import gc

import torch as t
from tqdm import tqdm

from connectivity.attribution import y_effect, __old_get_effect, get_edge_attr_feature
from utils.activation import get_is_tuple, get_hidden_states, get_hidden_attr
from utils.sparse_coo_helper import rearrange_weights, aggregate_weights, aggregate_nodes
from utils.graph_utils import topological_sort
from utils.ablation_fns import id_ablation

def get_circuit(
    clean,
    patch,
    model,
    dictionaries,
    metric_fn,
    embed,
    resids,
    attns=None,
    mlps=None,
    metric_kwargs=dict(),
    ablation_fn=None,
    aggregation='sum', # or 'none' for not aggregating across sequence position
    node_threshold=0.1,
    edge_threshold=0.01,
    steps=10,
    nodes_only=False,
    dump_all=False,
    save_path=None,
):
    return __old__get_circuit_feature_resid_only(
        clean,
        patch,
        model,
        embed,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        ablation_fn=ablation_fn,
        normalise_edges=False, # old test, not used anymore
        use_start_at_layer=False,
        aggregation=aggregation,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        nodes_only=nodes_only,
        steps=steps,
        dump_all=dump_all,
        save_path=save_path,
    )

# TODO : remove this function. Should be a special case of get_circuit_feature
def __old__get_circuit_feature_resid_only(
        clean,
        patch,
        model,
        embed,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
        ablation_fn=None,
        normalise_edges=False, # whether to divide the edges entering a node by their sum
        use_start_at_layer=False, # Whether to compute the layer-wise effects with the start at layer argument to save computation
        aggregation='max', # or 'none' for not aggregating across sequence position
        node_threshold=0.1,
        edge_threshold=0.01,
        nodes_only=False,
        steps=10,
        dump_all=False,
        save_path=None,
):
    if dump_all and save_path is None:
        raise ValueError("If dump_all is True, save_path must be provided.")
    
    all_submods = [embed] + [submod for submod in resids]
    last_layer = resids[-1]
    n_layers = len(resids)
    
    # get the shape of the hidden states
    is_tuple = get_is_tuple(model, all_submods)

    if use_start_at_layer:
        raise NotImplementedError
    
    # get encoding and reconstruction errors for clean and patch
    hidden_states_clean = get_hidden_states(model, submods=all_submods, dictionaries=dictionaries, is_tuple=is_tuple, input=clean)

    if patch is None:
        patch = clean
    
    hidden_states_patch = get_hidden_states(model, submods=all_submods, dictionaries=dictionaries, is_tuple=is_tuple, input=patch)
    hidden_states_patch = {k : ablation_fn(v) for k, v in hidden_states_patch.items()}
    
    features_by_submod = {}

    # start by the effect of the last layer to the metric
    edge_effect, nodes_attr = y_effect(
        model,
        clean, hidden_states_clean, hidden_states_patch,
        last_layer, all_submods,
        dictionaries, is_tuple,
        steps, metric_fn, metric_kwargs,
        normalise_edges, node_threshold, edge_threshold,
        features_by_submod
    )
    nodes = {}
    # print(f'resid_{len(resids)-1}')
    nodes[f'resid_{len(resids)-1}'] = nodes_attr[last_layer]

    if nodes_only:
        for layer in reversed(range(n_layers)):
            if layer > 0:
                # print(f'resid_{layer-1}')
                nodes[f'resid_{layer-1}'] = nodes_attr[resids[layer-1]]
            else:
                # print('embed')
                nodes['embed'] = nodes_attr[embed]

        print(nodes.keys())
        return nodes, {}
    
    edges = {}
    edges[f'resid_{len(resids)-1}'] = {'y' : edge_effect}
    
    # Now, backward through the model to get the effects of each layer on its successor.
    for layer in reversed(range(n_layers)):
        # print("Layer", layer, "threshold", edge_threshold)
        # print("Number of downstream features:", len(features_by_submod[resids[layer]]))
        
        resid = resids[layer]
        if layer > 0:
            prev_resid = resids[layer-1]
        else:
            prev_resid = embed

        RR_effect = __old_get_effect(
            model,
            clean, hidden_states_clean, hidden_states_patch,
            dictionaries,
            layer, prev_resid, resid,
            features_by_submod,
            is_tuple, steps, normalise_edges,
            nodes_attr, node_threshold, edge_threshold,
        )
    
        if layer > 0:
            nodes[f'resid_{layer-1}'] = nodes_attr[prev_resid]
            edges[f'resid_{layer-1}'][f'resid_{layer}'] = RR_effect
        else:
            nodes['embed'] = nodes_attr[prev_resid]
            edges['embed'][f'resid_0'] = RR_effect
        
        gc.collect()
        t.cuda.empty_cache()

    rearrange_weights(nodes, edges)
    aggregate_weights(nodes, edges, aggregation, dump_all=dump_all, save_path=save_path)

    return nodes, edges

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
                           # Method to aggregate the edge weights across sequence positions and batch elements.
                           # Supported methods are 'sum' and 'max'.
        threshold=0.01, # float. Threshold for edge pruning.
                             # When all edges leaving a node have a weight below this threshold, the node is pruned
                             # and the result is equivalent to the node being below the 'node_threshold'.
        edge_circuit=True, # bool. Whether to compute edge attribution or node attribution.
        steps=10, # int. Number of steps for the integrated gradients.
                  # When this value equals 1, only one gradient step is computed and the method is equivalent to
                  # the Attribution Patching's paper method.
):
    """
    When feature dictionaries are not partitioned (~ endowed with the discrete partition)
    it would be stupid to actually consider the discrete partition. This implementation is faster.
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
        for downstream in tqdm(topological_order[::-1]):
            upstreams = architectural_graph[downstream]
            if upstreams == [] or upstreams is None:
                continue

            edges[downstream] = get_edge_attr_feature(
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
            ) # return a dict of upstream -> effect
        
        # Now, aggregate the weights across sequence positions and batch elements.
        shapes = {k.name : hidden_states_clean[k.name].act.shape for k in all_submods}

        rearrange_weights(shapes, edges)
        aggregate_weights(shapes, edges, aggregation)

        return edges
    
    else:
        # Do node attribution.
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
        aggregate_nodes(nodes, aggregation)
        
        return nodes

def get_circuit_roi():
    """
    When feature dictionaries are endowed with arbitrary partitions, each part is a region of interest
    or a node in the computational graph.
    """
    pass