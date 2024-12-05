import torch

from utils.activation import get_hidden_states, get_is_tuple
from utils.graph_utils import topological_sort

@torch.no_grad()
def run_graph(
        model, # UnifiedTransformer (from nnsight)
        architectural_graph, # dict of downstream -> upstream (str -> str)
                             # The architectural graph of the model representing dependencies between modules.
                             # The sink is 'y' and all necessary modules and connections should be included in this graph.
                             # In this dict, modules are represented by their names (str).
        name2mod, # dict of str -> Submod
        dictionaries, # dict of str -> SAE
                      # The feature dictionaries to use for the interpretation.
                      # Should be at least one for every module appearing in the architecture graph.
        clean, # tensor (batch, seq_len) : the input to the model
        patch, # tensor (batch, seq_len) or None : the counterfactual input to the model to ablate edges
        mask, # dict of module -> SparseAct (str -> SparseAct)
              # The masks to apply to the hidden states of the model.
              # Used to ablate nodes outside the circuit under consideration.
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

    if patch is None: patch = clean
    is_tuple = get_is_tuple(model, all_submods)
    patch_states = get_hidden_states(
        model, submods=all_submods, dictionaries=dictionaries, is_tuple=is_tuple, input=patch
    )
    patch_states = {k : ablation_fn(v) for k, v in patch_states.items()}

    topological_order = topological_sort(architectural_graph)
    
    ##########
    # Forward pass with ablation :
    # Intervene on the forward pass by ablating undesired nodes.
    ##########

    with model.trace(clean), torch.no_grad():
        for submod in topological_order:
            if submod == 'y': continue
            # Get intermediate hidden states
            dictionary = dictionaries[submod]
            submod_nodes = mask[submod].clone()
            x = name2mod[submod].module.output
            if is_tuple[submod]:
                x = x[0]
            
            # go from latent space to feature space
            f = dictionary.encode(x)
            res = x - dictionary(x)

            # ablate features outside the circuit
            if complement: submod_nodes = ~submod_nodes
            submod_nodes.resc = submod_nodes.resc.expand(*submod_nodes.resc.shape[:-1], res.shape[-1])

            f[...,~submod_nodes.act] = patch_states[submod].act[...,~submod_nodes.act]
            res[...,~submod_nodes.resc] = patch_states[submod].res[...,~submod_nodes.resc]
            
            # go back to latent space
            if is_tuple[submod]:
                name2mod[submod].module.output[0][:] = dictionary.decode(f) + res
            else:
                name2mod[submod].module.output = dictionary.decode(f) + res

        # Compute the final metrics
        if isinstance(metric_fn, dict):
            metric = {}
            for name, fn in metric_fn.items():
                met = fn(model, metric_fn_kwargs).save()
                metric[name] = met
        else:
            raise ValueError("metric_fn must be a dict of functions")

    return metric
