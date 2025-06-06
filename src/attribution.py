import torch

from utils.activation import SparseAct, get_is_tuple, get_hidden_states, get_hidden_attr
from utils.sparse_coo_helper import rearrange_weights, aggregate_weights, aggregate_nodes
from utils.graph_utils import topological_sort
from utils.ablation_fns import id_ablation

DEBUGGING = False

if DEBUGGING:
    tracer_kwargs = {'validate' : True, 'scan' : True}
else:
    tracer_kwargs = {'validate' : False, 'scan' : False}

def _get_edge_attr_feature(
        model, # Unified transformer (from nnsight)
               # model to interpret
        clean, # Tokenized clean input
        hidden_states_clean, # dict of str -> SparseAct
                             # The hidden states in feature space.
        hidden_states_patch, # dict of str -> SparseAct
        dictionaries, # dict of str -> SAE
                      # The feature dictionaries to use for the interpretation.
                      # Should be at least one for every module appearing in the architecture graph.
        downstream, # str : downstream module
        upstreams, # list of str : modules upstream of downstream
        name2mod, # dict str -> Submodule : name to module
        features_by_submod, # dict of str -> list of int
                            # Features of interest for each module.
        is_tuple, # dict of str -> bool
                  # Whether the module output is a tuple or not.
        steps, # int. Number of steps for the integrated gradients.
                  # When this value equals 1, only one gradient step is computed and the method is equivalent to
                  # the Attribution Patching's paper method.
        edge_threshold, # float. Threshold for pruning.
        metric_fn, metric_kwargs=dict(), # dict
                              # Additional arguments to pass to the metric function.
                              # e.g. if the metric function is the logit, one should add the position of the target logit.
        restriction=None, # dict of str -> bool SparseAct. Restriction on the features to consider for each module. If None, all features are considered.
    ):
    """
    Helper function for get_circuit_feature.
    Get the effect of some upstream modules on some downstream module. Uses integrated gradient attribution.
    Do not support feature partitions (ROIs). See get_edge_attr_roi for that.
    """
    try:
        downstream_features = features_by_submod[downstream]
    except KeyError:
        raise ValueError(f"Module {downstream} has no features to compute effects for")

    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'cfg'):
        device = model.cfg.device
    else:
        raise ValueError("Can't get model device :c")

    #print(f"Computing effects for layer {layer} with {len(downstream_features)} features")

    if not features_by_submod[downstream]: # handle empty list
        raise ValueError(f"Module {downstream} has no features to compute effects for")
    
    downstream_submod = name2mod[downstream]

    # used for resid layers. If upstreams include the current layer's attn and mlp, do not do the full forward pass
    # In that case, resid_{l+1} = attn_{l+1} + mlp_{l+1} + resid_{l} (or upstreams of resid_{l} if the current
    #                                                                 settings do not include resid_{l}).
    # Otherwise, resid_{l+1} = f(resid_{l})
    do_the_forward = True
    for up in upstreams:
        if up != 'embed' and not downstream == 'y':
            u_typ, u_l = up.split('_')
            d_typ, d_l = downstream.split('_')
            if u_l == d_l:
                do_the_forward = False
                break
    
    # dicts upstream_name -> downstream_feature -> effect_indices / effect_values
    # Store information to create a sparse coo tensor storing the attribution of upstream -> downstream
    effect_indices = {}
    effect_values = {}

    # For all downstream feat, reconstruct the input, do a forward pass
    # in the module, patch only the feature of interest, and compute the effect on the metric.
    for downstream_feat in downstream_features:
        metrics = []
        fs = {}

        for step in range(1, steps+1):
            alpha = step / steps

            reconstructed_input = 0

            for upstream_name in upstreams:
                # get the hidden states of the upstream module in feature space
                # Linearly interpolate between the clean and patched hidden states
                upstream_act = alpha * hidden_states_clean[upstream_name] + (1-alpha) * hidden_states_patch[upstream_name] # act shape (batch_size, seq_len, n_features) res shape (batch_size, seq_len, d_model)
                upstream_act.act = upstream_act.act.clone().detach()
                upstream_act.act.requires_grad_().retain_grad()
                upstream_act.res = upstream_act.res.clone().detach()
                upstream_act.res.requires_grad_().retain_grad()

                # Keep track of that state for later gradients
                if fs.get(upstream_name) is None:
                    fs[upstream_name] = []
                fs[upstream_name].append(upstream_act)
                
                # go back to latent space
                up_out = dictionaries[upstream_name].decode(upstream_act.act) + upstream_act.res
                # apply ln_post if needed (otherwise, this will be a torch.nn.Identity)
                reconstructed_input += name2mod[upstream_name].LN_post.forward(up_out)
            
            if downstream != 'y':
                # get the output of the downstream module.
                # Most of the times, attn and mlp are preceded by a LN layer.
                # For modules without a preceding LN layer, this will be a torch.nn.Identity.
                ln = downstream_submod.LN_pre.forward(reconstructed_input)
                if 'attn' in downstream:
                    y = downstream_submod.module.forward(ln, ln, ln)
                elif 'mlp' in downstream:
                    y = downstream_submod.module.forward(ln)
                elif 'resid' in downstream:
                    if do_the_forward:
                        y = downstream_submod.module.forward(ln)
                    else:
                        y = ln
                else:
                    raise ValueError(f"Downstream module {downstream} not recognized")
                
                # TODO : check that with autograd and all these interventions, the gradients are properly computed
                # Go to feature space
                g = dictionaries[downstream].encode(y)
                y_hat = dictionaries[downstream].decode(g)
                y_res = y - y_hat
                downstream_act = SparseAct(
                    act=g,
                    res=y_res
                )
                
                # In feature space, patch only the feature of interest before proceeding to the rest of the forward pass to get the effect on the metric.
                # downstream_feat is the flattened index across sequence and features. Extract the feature index and the sequence index :
                feat_idx = downstream_feat % (downstream_act.act.size(-1) + 1)
                seq_idx = downstream_feat // (downstream_act.act.size(-1) + 1)
                # deal with the reconstruction error node :
                if feat_idx == downstream_act.act.size(-1):
                    diff = downstream_act.res[..., seq_idx, :] - hidden_states_clean[downstream].res[..., seq_idx, :]
                else:
                    # Get the scalar value (coeff) of the downstream feature
                    diff = downstream_act.act[..., seq_idx, feat_idx] - hidden_states_clean[downstream].act[..., seq_idx, feat_idx]
                    # Use this to scale the target feature vector
                    diff = diff * dictionaries[downstream].W_dec[feat_idx]
            
                # Do the forward pass with that patch and get the effect on the metric
                with model.trace(clean, **tracer_kwargs):
                    if is_tuple[downstream]:
                        downstream_submod.module.output[0][..., seq_idx, :] += diff # downstream_act - clean_states Remove the current feature's effect and add it's patched effect
                    else:
                        downstream_submod.module.output[..., seq_idx, :] += diff

                    metrics.append(metric_fn(model, metric_kwargs).save())
            
            # If downstream == 'y', there is no "feature" of interest, only the whole output.
            else:
                with model.trace(clean, **tracer_kwargs):
                    downstream_submod.module.output = reconstructed_input
                    metrics.append(metric_fn(model, metric_kwargs).save())

        # Backward pass...
        metric = sum([m for m in metrics])
        metric.sum().backward(retain_graph=True)

        # ... to get the effect as grad @ delta
        for upstream_name in upstreams:
            mean_act_grad = sum([f.act.grad for f in fs[upstream_name]]) / steps
            mean_res_grad = sum([f.res.grad for f in fs[upstream_name]]) / steps

            grad = SparseAct(act=mean_act_grad, res=mean_res_grad)
            delta = (hidden_states_patch[upstream_name] - hidden_states_clean[upstream_name]).detach()

            if effect_indices.get(upstream_name) is None:
                effect_indices[upstream_name] = {}
                effect_values[upstream_name] = {}

            effect = (grad @ delta).abs().to_tensor().flatten()

            # Store the effect as a sparse tensor
            effect_indices[upstream_name][downstream_feat] = torch.where(
                effect.abs() > edge_threshold,
                effect,
                torch.zeros_like(effect)
            ).nonzero().squeeze(-1)

            # Restrict indices to available features :
            if restriction is not None:
                print(restriction[upstream_name].act.size())
                raise NotImplementedError("Restriction not implemented yet")
                mask = restriction[upstream_name].to_tensor() # (n_features + 1)
                temp = effect_indices[upstream_name][downstream_feat]
                effect_indices[upstream_name][downstream_feat] = temp[mask[temp % mask.size(0)]]

            effect_values[upstream_name][downstream_feat] = effect[effect_indices[upstream_name][downstream_feat]]

    # get shapes for the return sparse tensors
    d_downstream_contracted = torch.tensor(hidden_states_clean[downstream].act.size() if downstream != 'y' else (0,)) # if downstream == 'y', then the output is a single scalar
    d_downstream_contracted[-1] += 1
    d_downstream_contracted = d_downstream_contracted.prod()

    d_upstream_contracted = {}
    for upstream_name in upstreams:
        d_upstream_contracted[upstream_name] = torch.tensor(hidden_states_clean[upstream_name].act.size())
        d_upstream_contracted[upstream_name][-1] += 1
        d_upstream_contracted[upstream_name] = d_upstream_contracted[upstream_name].prod()

    # Create the sparse_coo_tensor containing the effect of each upstream module on the downstream module
    effects = {}
    for upstream_name in upstreams:
        # converts the dictionary of indices to a tensor of indices
        effect_indices[upstream_name] = torch.tensor(
            [[downstream_feat for downstream_feat in downstream_features for _ in effect_indices[upstream_name][downstream_feat]],
            torch.cat([effect_indices[upstream_name][downstream_feat] for downstream_feat in downstream_features], dim=0)]
        ).to(device).long()
        effect_values[upstream_name] = torch.cat([effect_values[upstream_name][downstream_feat] for downstream_feat in downstream_features], dim=0)

        potential_upstream_features = effect_indices[upstream_name][1] # list of indices
        
        if features_by_submod.get(upstream_name) is None:
            features_by_submod[upstream_name] = potential_upstream_features.unique().tolist()
        else:
            features_by_submod[upstream_name] += potential_upstream_features.unique().tolist()
            features_by_submod[upstream_name] = list(set(features_by_submod[upstream_name]))

        effects[upstream_name] = torch.sparse_coo_tensor(
            effect_indices[upstream_name], effect_values[upstream_name],
            (d_downstream_contracted, d_upstream_contracted[upstream_name])
        )

    return effects

def _get_edge_attr_roi():
    """
    TODO
    Helper function for get_circuit_roi.
    """
    pass

def get_circuit_feature(
        clean, # Tokenized clean input
        patch, # Tokenized patch input (optional)
        model, # Unified transformer (from nnsight)
               # model to interpret
        architectural_graph, # dict of downstream -> upstream modules.
                             # The architecture graph of model.
                             # The sink is 'y' and all necessary modules and connections should be included in this graph.
                             # In this dict, modules are represented by their names (str).
        name2mod, # dict str -> Submod
        dictionaries, # dict of str -> SAE
                      # The feature dictionaries to use for the interpretation.
                      # Should be at least one for every module appearing in the architecture graph.
        metric_fn, # callable
                   # Scalar metric function representing what we want to measure from the model.
                   # e.g. if one wants to make a computational graph, the target logits can be used.
                   #      if one wants to compute the circuit for responsible for a specific feature / probe / ... in
                   #      the hidden states, the activation of that feature (resp. the output of the probe) can be used.
        metric_kwargs=dict(), # dict
                              # Additional arguments to pass to the metric function.
                              # e.g. if the metric function is the logit, one should add the position of the target logit.
        ablation_fn=id_ablation, # callable
                          # Ablation function used for integrated gradient. Applied to the patched hidden states.
                          # The results gives the baseline for the integrated gradients.
        aggregation='sum', # str
                           # Method to aggregate the weights across sequence positions and batch elements.
                           # Supported methods are 'sum' and 'max'.
        threshold=0.01, # float. Threshold for pruning.
        edge_circuit=True, # bool. Whether to compute edge attribution or node attribution.
        steps=10, # int. Number of steps for the integrated gradients.
                  # When this value equals 1, only one gradient step is computed and the method is equivalent to
                  # the Attribution Patching's paper method.
        restriction=None, # dict of str -> bool SparseAct. Restriction on the features to consider for each module. If None, all features are considered.
        freq=None, # dict. Frequency of feature usage in the circuit.
):
    """
    Run circuit discovery at feature level.
    """
    # gather all modules involved in the computation : start from 'y' and do a reachability search.
    visited = set()
    to_visit = ['y']
    while to_visit:
        downstream = to_visit.pop()
        if downstream in visited:
            continue
        visited.add(downstream)
        to_visit += architectural_graph[downstream]
        
    all_submods = list(visited)
    all_submods.remove('y')
    all_submods = [name2mod[name] for name in all_submods]
    
    # get the shape of the hidden states
    is_tuple = get_is_tuple(model, all_submods)

    # get hidden states for clean and patch inputs
    hidden_states_clean = get_hidden_states(
        model, submods=all_submods, dictionaries=dictionaries, is_tuple=is_tuple, input=clean
    )

    if patch is None:
        patch = clean

    hidden_states_patch = get_hidden_states(
        model, submods=all_submods, dictionaries=dictionaries, is_tuple=is_tuple, input=patch
    )
    hidden_states_patch = {k : ablation_fn(v) for k, v in hidden_states_patch.items()}

    if edge_circuit:
        # Do edge attribution. Go through features of interest in reverse topological order
        # to know which of their ancestors contribute and how much.
        features_by_submod = {'y' : [0]}
        
        # Backward through the graph to get effects of upstream modules on downstream modules. Start from 'y'.
        topological_order = topological_sort(architectural_graph)

        edges = {}
        for downstream in topological_order[::-1]:
            upstreams = architectural_graph[downstream]
            if upstreams == [] or upstreams is None:
                continue

            edges[downstream] = _get_edge_attr_feature(
                model,
                clean, hidden_states_clean, hidden_states_patch,
                dictionaries,
                downstream,
                architectural_graph[downstream],
                name2mod,
                features_by_submod,
                is_tuple, steps,
                threshold,
                metric_fn, metric_kwargs=metric_kwargs,
                restriction=restriction,
            ) # return a dict of upstream -> effect
        
        # Now, aggregate the weights across sequence positions and batch elements.
        shapes = {k.name : hidden_states_clean[k.name].act.shape for k in all_submods}

        rearrange_weights(shapes, edges)
        aggregate_weights(shapes, edges, aggregation, freq=freq)

        return edges
    
    else:
        # Do classic node attribution.
        nodes = get_hidden_attr(
            model,
            all_submods,
            dictionaries,
            is_tuple,
            clean,
            metric_fn,
            patch,
            ablation_fn,
            steps,
            metric_kwargs,
        )
        aggregate_nodes(nodes, aggregation, freq=freq, threshold=threshold)
        
        return nodes

def get_circuit_roi():
    """
    When feature dictionaries are endowed with arbitrary partitions, each part is a region of interest
    or a node in the computational graph.
    """
    pass