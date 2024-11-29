import torch

from utils.activation import SparseAct, get_hidden_states, get_is_tuple
from utils.graph_utils import topological_sort

def __old_reorder_mask(edges):
    """
    mask : dict of dict of sparse_coo tensors
    returns a dict of dict of sparse_coo tensors
    """
    new_mask = {}
    for up in edges:
        for down in edges[up]:
            if down not in new_mask:
                new_mask[down] = {}
            new_mask[down][up] = edges[up][down]
    return new_mask

@torch.jit.script
def compiled_loop_pot_ali(mask_idx, potentially_alive, up_nz):
    # in the mask indices has shape (2, ...), potentially alive downstream features are indices in [0] st indices[1] is in up_nz
    for i, idx in enumerate(mask_idx[1]):
        if up_nz[idx]:
            potentially_alive[mask_idx[0][i]] = True

@torch.no_grad()
def run_graph(
        model, # UnifiedTransformer (from nnsight)
        computational_graph, # dict of downstream -> upstream -> edge_mask (str -> str -> bool sparse_coo_tensor)
                             # The computational graph of the model.
                             # The sink is 'y' and all necessary modules and connections should be included in this graph.
                             # In this dict, modules are represented by their names (str).
        name2mod, # dict of str -> Submod
        dictionaries, # dict of str -> SAE
                      # The feature dictionaries to use for the interpretation.
                      # Should be at least one for every module appearing in the architecture graph.
        clean, # tensor (batch, seq_len) : the input to the model
        patch, # tensor (batch, seq_len) or None : the counterfactual input to the model to ablate edges
        metric_fn, # dict of str -> callable : scalar functions to evaluate the model
        metric_fn_kwargs, # dict
                          # Additional arguments to pass to the metric function.
                          # e.g. if the metric function is the logit, one should add the position of the target logit.
        ablation_fn, # callable : the function used to get the patched states. Applied to the hidden states from the forward pass on the patch input.
        complement=False, # bool : whether to use the given graph or its complement
):
    ##########
    # Initialization :
    # - get all necessary objects, lists and hidden states
    # - get the order in which to compute the modules
    ##########

    # create the architectural graph (dict of downstream -> upstream) from the computational graph (dict of downstream -> upstream -> edge_mask)
    architectural_graph = {}
    for downstream in computational_graph:
        architectural_graph[downstream] = list(computational_graph[downstream].keys())

    # gather all modules involved in the computation : start from 'y' and do a reachability search.
    visited = set()
    to_visit = ['y']
    while to_visit:
        downstream = to_visit.pop()
        if downstream in visited:
            continue
        visited.add(downstream)
        to_visit += architectural_graph.get(downstream, [])
        
    all_submods = list(visited)
    all_submods.remove('y')
    all_submods = [name2mod[name] for name in all_submods]
    
    # get the shape of the hidden states
    is_tuple = get_is_tuple(model, all_submods + [name2mod['y']])

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

    if complement:
        hidden_states_clean, hidden_states_patch = hidden_states_patch, hidden_states_clean

    # Compute the topological order of the graph to know in which order to compute the modules.
    topological_order = topological_sort(architectural_graph)

    ##########
    # Forward pass with ablation :
    # - compute the new hidden states of the model in the correct order.
    # - Modules without upstreams are left unchanged. They are the "input" modules. It is possible that you don't want to start at
    #   the embed layer, it means that we start modifying the forward pass only for later layers.
    # - compute each downstream feature with a forward pass on the masked input.
    ##########

    def _aux_reconstruct_input(downstream, f_, upstreams):
        reconstructed_input = 0
        for upstream in upstreams:
            up_clean = hidden_states[upstream]
            up_patch = hidden_states_patch[upstream]
            mask = computational_graph[downstream][upstream][f_].to_dense() if downstream != 'y' else computational_graph[downstream][upstream].to_dense() # shape (f_up + 1)

            up_masked = SparseAct(
                act = up_patch.act.clone(),# * (1 - mask[:-1]) + up_clean.act * mask[:-1], # TODO : test this
                res = up_clean.res.clone() if mask[-1] else up_patch.res.clone()
            )
            up_masked.act[:, :, mask[:-1]] = up_clean.act[:, :, mask[:-1]]

            reconstructed_input += dictionaries[upstream].decode(up_masked.act) + up_masked.res
        return reconstructed_input

    hidden_states = {}
    for downstream in topological_order:
        upstreams = architectural_graph.get(downstream, None)
        if upstreams is None or upstreams == []:
            # This module has no upstream, so it is an input module.
            hidden_states[downstream] = hidden_states_clean[downstream]
            continue

        # get the hidden states of the upstream modules
        
        if downstream == 'y':
            hidden_states['y'] = _aux_reconstruct_input(downstream, 0, upstreams)
            continue

        f = hidden_states_patch[downstream].act.clone()
        res = hidden_states_patch[downstream].res.clone()
        potentially_alive = torch.zeros(f.shape[-1] + 1, device=f.device, dtype=torch.bool)

        for upstream in architectural_graph[downstream]:
            # We want to know if inputs to target feature have been modified w.r.t. the baseline input.
            # This will tell us which target feature we need to recompute separately.
            mask = computational_graph[downstream][upstream] # shape (f_down + 1, f_up + 1)
            up_state = hidden_states[upstream].act - hidden_states_patch[upstream].act # shape (batch, seq_len, f_up)
            # reduce to (f_up,) by maxing over batch and seq_len (should only have positive entries, but a .abs() can't hurt)
            up_state = up_state.abs().amax(dim=(0, 1)) # shape (f_up)
            up_nz = torch.cat([up_state > 1e-6, torch.tensor([True], device=f.device)]) # shape (f_up + 1). Always keep the res feature alive
            
            # print("Number of potentially alive features upstream : ", up_nz.sum().item())
            compiled_loop_pot_ali(mask.indices(), potentially_alive, up_nz)
        
        potentially_alive = potentially_alive.nonzero().squeeze(1)

        # used for resid layers. If upstreams include the current layer's attn and mlp, do not do the full forward pass
        do_the_forward = True
        for up in upstreams:
            if up != 'embed' and not downstream == 'y':
                u_typ, u_l = up.split('_')
                d_typ, d_l = downstream.split('_')
                if u_l == d_l:
                    do_the_forward = False
                    break

        # f is currently filled with baseline states.
        # Fill in the potentially alive features with new values.
        for f_ in potentially_alive:
            # Recreate the input to the downstream module by keepeing only those relevant for the current feature f_.
            reconstructed_input = _aux_reconstruct_input(downstream, f_, upstreams)

            down_mod = name2mod[downstream]

            ln = down_mod.LN.forward(reconstructed_input) if down_mod.LN is not None else reconstructed_input
            if 'attn' in downstream:
                y = down_mod.module.forward(ln, ln, ln)
            elif 'mlp' in downstream:
                y = down_mod.module.forward(ln)
            elif 'resid' in downstream:
                # If the current layer has a resid and it's respective attn & mlp, it is just the sum of the previous resid and both of those.
                # Otherwise, the input has to go through the whole transformer block.
                # This would be much easier if I could give an identity module to the submod class, but then I could not intervene
                # in the .output later on. To do that I would have to properly incorporate that identity module inside of the architecture of the
                # transformer so that nnsight knows how to deal with it.
                if do_the_forward:
                    y = down_mod.module.forward(ln)
                else:
                    y = ln
            else:
                raise ValueError(f"Downstream module {downstream} not recognized")
            
            if f_ < f.shape[-1]:
                f[..., f_] = dictionaries[downstream].encode(y)[..., f_]
            else:
                res = y - dictionaries[downstream](y)

        hidden_states[downstream] = SparseAct(act=f, res=res)

    ##########
    # Compute the final metrics :
    ##########
    with model.trace(clean):
        name2mod['y'].module.output = hidden_states['y']
        
        metric = {}
        for name, fn in metric_fn.items():
            met = fn(model, metric_fn_kwargs).save()
            metric[name] = met
        
    return metric

@torch.no_grad()
def __old_run_graph(
        model,
        submodules,
        dictionaries,
        mod2name,
        clean,
        patch,
        mask,
        metric_fn,
        metric_fn_kwargs,
        ablation_fn,
        complement=False,
    ):
    """
    model : nnsight model
    submodules : list of model submodules
        Should be ordered by appearance in the sequencial model
    dictionaries : dict
        dict [submodule] -> SAE
    name_dict : dict
        dict [submodule] -> str
    clean : str, list of str or tensor (batch, seq_len)
        the input to the model
    patch : None, str, list of str or tensor (batch, seq_len)
        the counterfactual input to the model to ablate edges
    mask : edges or tuple(nodes, edges)
        the graph to use. Accept any type of sparse_coo tensor, but preferably bool.
        Any index in these tensors will be used as an edge.
    metric_fn : callable
        the function to evaluate the model.
        It can be CE, accuracy, logit for target token, etc.
    metric_fn_kwargs : dict
        the kwargs to pass to metric_fn. E.g. target token.
    ablation_fn : callable
        the function used to get the patched states.
        E.g. : mean ablation means across batch and sequence length or only across one of them.
               zero ablation : patch states are just zeros
               id ablation : patch states are computed from the patch input and left unchanged

    returns the metric on the model with the graph
    """

    # various initializations
    if isinstance(mask, tuple):
        mask = mask[1]
    graph = ... # reorder_mask(mask)

    if complement:
        raise NotImplementedError("Complement is not implemented yet")

    name2mod = {v : k for k, v in mod2name.items()}
    
    is_tuple = {}
    input_is_tuple = {}
    with model.trace("_"), torch.no_grad():
        for submodule in submodules:
            input_is_tuple[submodule] = type(submodule.input.shape) == tuple
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    if patch is None:
        patch = clean

    # get patch hidden states
    patch_states = get_hidden_states(model, submodules, dictionaries, is_tuple, patch)
    patch_states = {k : ablation_fn(v).clone() for k, v in patch_states.items()}

    # forward through the model by computing each node as described by the graph and not as the original model does

    # For each downstream module, get it's potentially alive features (heuristic to not compute one forward pass per node
    # as there are too many of them) by reachability from previously alive ones
    # Then, for each of these features, get it's masked input, and compute a forward pass to get this particular feature.
    # This gives the new state for this downstream output.

    # with model.trace(clean):
    hidden_states = {}
    for downstream in submodules:
        # get downstream dict, output, ...
        downstream_dict = dictionaries[downstream]
        down_name = mod2name[downstream]
        # print(f"Computing {down_name}")
        # TOD? : this surely can be replaced by a single call to trace, or none at all
        with model.trace(clean):
            if input_is_tuple[downstream]:
                input_shape = downstream.input[0].shape
                input_dict = downstream.input[1:].save()
            else:
                input_shape = downstream.input.shape
            
            x = downstream.output
            if is_tuple[downstream]:
                x = x[0]
            x = x.save()
        
        if isinstance(input_shape, tuple):
            input_shape = input_shape[0]
            
        x_hat, f = downstream_dict(x, output_features=True)
        res = x - x_hat

        # print("Got x_hat and f")

        # if downstream is embed, there is no upstream and the result stays unchanged
        if down_name == 'embed' or downstream == submodules[0]:
            # print("Embed or first layer")
            hidden_states[downstream] = SparseAct(act=f, res=res)
            continue

        # otherwise, we have to do the computation as the graph describes it, and each downstream
        # feature, including the res, is computed from a different set of upstream features

        potentially_alive = torch.zeros(f.shape[-1] + 1, device=f.device, dtype=torch.bool)

        for up_name in graph[down_name]:
            upstream = name2mod[up_name]
            if upstream not in submodules:
                continue
            mask = graph[down_name][up_name] # shape (f_down + 1, f_up + 1)

            upstream_hidden = hidden_states[upstream].act # shape (batch, seq_len, f_up)
            # reduce to (f_up) by maxing over batch and seq_len (should only have positive entries, but a .abs() can't hurt)
            upstream_hidden = upstream_hidden.abs().amax(dim=(0, 1)) # shape (f_up)
            up_nz = torch.cat([upstream_hidden > 0, torch.tensor([True], device=f.device)]) # shape (f_up + 1). Always keep the res feature alive
            
            # print("Number of potentially alive features upstream : ", up_nz.sum().item())
            ... # compiled_loop_pot_ali(mask.indices(), potentially_alive, up_nz)

        potentially_alive = potentially_alive.nonzero().squeeze(1)
        # print("Number of potentially alive features downstream : ", potentially_alive.size(0))

        f[...] = patch_states[downstream].act # shape (batch, seq_len, f_down)
        for f_ in potentially_alive:
            edge_ablated_input = torch.zeros(tuple(input_shape)).to(f.device)
            for up_name in graph[down_name]:
                upstream = name2mod[up_name]
                if upstream not in submodules:
                    continue
                upstream_dict = dictionaries[upstream]
                
                mask = graph[down_name][up_name][f_].to_dense() # shape (f_up + 1)

                edge_ablated_upstream = SparseAct(
                    act = patch_states[upstream].act,
                    res = hidden_states[upstream].res if mask[-1] else patch_states[upstream].res
                )
                edge_ablated_upstream.act[:, :, mask[:-1]] = hidden_states[upstream].act[:, :, mask[:-1]]

                edge_ablated_input += upstream_dict.decode(edge_ablated_upstream.act) + edge_ablated_upstream.res

            module_type = down_name.split('_')[0]
            # TODO : batch these forward passes to speed up the process
            if module_type == 'resid':
                # if resid only, do this, othewise, should be literally the identity as the sum gives resid_post already.
                if input_is_tuple[downstream]:
                    edge_ablated_out = downstream.forward(edge_ablated_input, **input_dict.value[0])
                else:
                    edge_ablated_out = downstream.forward(edge_ablated_input)
                if is_tuple[downstream]:
                    edge_ablated_out = edge_ablated_out[0]
            else:
                # if attn or mlp, use corresponding LN
                raise NotImplementedError(f"Support for module type {module_type} is not implemented yet")
            if f_ < f.shape[-1]:
                # TODO : add option in sae forward to get only one feature to fasten this
                #        only after testing that it works like this first and then checking that it
                #        is faster and doesn't break anything
                #        more generally, try to compress each node function as much as possible.
                f[..., f_] = downstream_dict.encode(edge_ablated_out)[..., f_] # replace by ", f_)"
            else:
                res = edge_ablated_out - downstream_dict(edge_ablated_out)

        hidden_states[downstream] = SparseAct(act=f, res=res)

        # if is_tuple[downstream]:
        #     downstream.output[0][:] = downstream_dict.decode(f) + res
        # else:
        #     downstream.output = downstream_dict.decode(f) + res

    last_layer = submodules[-1]
    with model.trace(clean):
        if is_tuple[last_layer]:
            last_layer.output[0][:] = dictionaries[last_layer].decode(hidden_states[last_layer].act) + hidden_states[last_layer].res
        else:
            last_layer.output = dictionaries[last_layer].decode(hidden_states[last_layer].act) + hidden_states[last_layer].res
        
        if isinstance(metric_fn, dict):
            metric = {}
            for name, fn in metric_fn.items():
                met = fn(model, metric_fn_kwargs).save()
                metric[name] = met
        else:
            raise ValueError("metric_fn must be a dict of functions")

    # for name, value in metric.items():
    #     value = value[torch.isfinite(value)]
    return metric
