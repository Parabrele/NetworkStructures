import torch

from utils.activation import get_hidden_states, get_is_tuple
from utils.graph_utils import topological_sort

@torch.no_grad()
def run_graph(
        model,
        architectural_graph,
        name2mod,
        dictionaries,
        clean,
        patch,
        mask,
        metric_fn,
        metric_fn_kwargs,
        ablation_fn,
        complement=False,
    ):
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
    
    with model.trace(clean), torch.no_grad():
        for submod in topological_order:
            if submod == 'y': continue
            dictionary = dictionaries[submod]
            submod_nodes = mask[submod].clone()
            x = name2mod[submod].module.output
            if is_tuple[submod]:
                x = x[0]
            f = dictionary.encode(x)
            res = x - dictionary(x)

            # ablate features
            if complement: submod_nodes = ~submod_nodes
            submod_nodes.resc = submod_nodes.resc.expand(*submod_nodes.resc.shape[:-1], res.shape[-1])

            f[...,~submod_nodes.act] = patch_states[submod].act[...,~submod_nodes.act]
            res[...,~submod_nodes.resc] = patch_states[submod].res[...,~submod_nodes.resc]
            
            if is_tuple[submod]:
                name2mod[submod].module.output[0][:] = dictionary.decode(f) + res
            else:
                name2mod[submod].module.output = dictionary.decode(f) + res

        if isinstance(metric_fn, dict):
            metric = {}
            for name, fn in metric_fn.items():
                met = fn(model, metric_fn_kwargs).save()
                metric[name] = met
        else:
            raise ValueError("metric_fn must be a dict of functions")

    return metric
