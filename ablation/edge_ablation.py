import torch

from utils.activation import SparseAct, get_hidden_states, get_is_tuple
from utils.graph_utils import topological_sort

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

    # Helper function to reconstruct the input to a downstream feature.
    def _aux_reconstruct_input(downstream, f_, upstreams):
        reconstructed_input = 0
        for upstream in upstreams:
            # get the hidden states of the upstream module in feature space
            up_clean = hidden_states[upstream]
            up_patch = hidden_states_patch[upstream]
            mask = computational_graph[downstream][upstream][f_].to_dense() if downstream != 'y' else computational_graph[downstream][upstream].to_dense() # shape (f_up + 1)

            # mask the hidden states to keep only inputs to the target downstream feature
            up_masked = SparseAct(
                act = up_patch.act.clone(),# * (1 - mask[:-1]) + up_clean.act * mask[:-1], # TODO : test this
                res = up_clean.res.clone() if mask[-1] else up_patch.res.clone()
            )
            up_masked.act[:, :, mask[:-1]] = up_clean.act[:, :, mask[:-1]]

            # go back to latent space
            up_out = dictionaries[upstream].decode(up_masked.act) + up_masked.res
            # apply ln_post if needed (otherwise, this will be a torch.nn.Identity)
            reconstructed_input += name2mod[upstream].LN_post.forward(up_out)
        return reconstructed_input

    hidden_states = {}
    for downstream in topological_order:
        upstreams = architectural_graph.get(downstream, None)
        if upstreams is None or upstreams == []:
            # This module has no upstream, so it is an input module.
            hidden_states[downstream] = hidden_states_clean[downstream]
            continue

        if downstream == 'y':
            # For the final output, the "hidden state" is just the aggregation of all that came before.
            hidden_states['y'] = _aux_reconstruct_input(downstream, 0, upstreams)
            continue

        f = hidden_states_patch[downstream].act.clone()
        res = hidden_states_patch[downstream].res.clone()
        potentially_alive = torch.zeros(f.shape[-1] + 1, device=f.device, dtype=torch.bool)

        for upstream in architectural_graph[downstream]:
            # We want to know if inputs to the target feature have been modified w.r.t. the baseline input.
            # This will tell us which target feature we need to recompute separately.
            # If dictionaries are sparse, only a small number of features will be potentially alive.
            mask = computational_graph[downstream][upstream] # shape (f_down + 1, f_up + 1)
            up_state = hidden_states[upstream].act - hidden_states_patch[upstream].act # shape (batch, seq_len, f_up)
            # reduce to (f_up,) by maxing over batch and seq_len (should only have positive entries, but a .abs() can't hurt)
            up_state = up_state.abs().amax(dim=(0, 1)) # shape (f_up)
            up_nz = torch.cat([up_state > 1e-6, torch.tensor([True], device=f.device)]) # shape (f_up + 1). Always keep the res feature alive
            
            # print("Number of potentially alive features upstream : ", up_nz.sum().item())
            compiled_loop_pot_ali(mask.indices(), potentially_alive, up_nz)
        
        potentially_alive = potentially_alive.nonzero().squeeze(1)

        # used for resid layers. If upstreams include the current layer's attn and mlp, do not do the full forward pass
        # In that case, resid_{l+1} = attn_{l+1} + mlp_{l+1} + resid_{l} (or upstreams of resid_{l} if the current
        #                                                                 settings do not include resid_{l}).
        # Otherwise, resid_{l+1} = resid_{l}
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
            # Recreate the input to the downstream module by keepeing only nodes relevant for the current feature f_.
            reconstructed_input = _aux_reconstruct_input(downstream, f_, upstreams)

            down_mod = name2mod[downstream]

            # Most of the times, attn and mlp are preceded by a LN layer.
            # For modules without a preceding LN layer, this will be a torch.nn.Identity.
            ln = down_mod.LN_pre.forward(reconstructed_input)
            if 'attn' in downstream:
                y = down_mod.module.forward(ln, ln, ln)
            elif 'mlp' in downstream:
                y = down_mod.module.forward(ln)
            elif 'resid' in downstream:
                if do_the_forward:
                    y = down_mod.module.forward(ln)
                else:
                    y = ln
            else:
                raise ValueError(f"Downstream module {downstream} not recognized. Must be 'attn_l', 'mlp_l' or 'resid_l'.")
            
            # Deal with the reconstruction error node :
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
