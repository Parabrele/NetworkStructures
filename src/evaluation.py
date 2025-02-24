import torch

from ablation.edge_ablation import run_graph as run_edge_ablation
from ablation.node_ablation import run_graph as run_node_ablation

from utils.activation import SparseAct
from utils.ablation_fns import mean_ablation, zero_ablation, id_ablation
from utils.graph_utils import get_mask, prune, get_n_nodes, get_n_edges, get_density

@torch.no_grad()
def faithfulness(
        model, # Unified transformer (from nnsight)
               # model to interpret
        name2mod, # dict str -> Submodule : name to module
        sae_dict, # dict of str -> SAE
                  # The feature dictionaries to use for the interpretation.
                  # Should be at least one for every module appearing in the architecture graph.
        clean, # Tokenized clean input
        circuit, # Output of some graph construction function
                 # Currently only get_circuit_feature is supported
                 # Contains attribution scores for nodes or edges
        architectural_graph, # dict of downstream -> upstream modules.
                             # The architecture graph of model.
                             # The sink is 'y' and all necessary modules and connections should be included in this graph.
                             # In this dict, modules are represented by their names (str).
        thresholds, # float or list of float to threshold `circuit`'s values.
        metric_fn, # dict of str -> callable : scalar functions to evaluate the model
        metric_fn_kwargs={}, # dict of additional arguments to pass to the metric function.
        patch=None, # Tokenized patch input (optional)
        ablation_fn=None, # callable
                          # Ablation function used for integrated gradient. Applied to the patched hidden states.
                          # The results gives the baseline for the integrated gradients.
        default_ablation='mean', # str : default ablation function to use if patch is None
        get_graph_info=True, # bool : whether to compute graph metrics (n_nodes, n_edges, avg_deg, density)
        node_ablation=False # bool : whether `circuit` contains node or edge attributions.
    ):
    """
    returns a dict
        threshold -> 'faithfulness'/'completeness' -> metric_name -> metric_value
        'complete' -> metric_name -> float (metric on the original model)
        'empty' -> float (metric on the fully ablated model, no edges)
    """
    if isinstance(thresholds, float):
        thresholds = [thresholds]
    if patch is not None and ablation_fn is None:
        ablation_fn = lambda x: x
    if patch is None and ablation_fn is None:
        if default_ablation == 'mean':
            ablation_fn = mean_ablation
        elif default_ablation == 'zero':
            ablation_fn = zero_ablation
        elif default_ablation == 'id':
            ablation_fn = id_ablation
        else:
            raise ValueError(f"Unknown default ablation function : {default_ablation}")
    
    if node_ablation:
        run_graph = run_node_ablation
    else:
        run_graph = run_edge_ablation
        
    results = {}

    ##########
    # Initialisation with useful values
    ##########

    # get unmodified logits
    with model.trace(clean):
        if isinstance(model.output, tuple):
            clean_logits = model.output[0]
        else:
            clean_logits = model.output
        clean_logits = clean_logits.save()
    
    # get metric on original model :
    metric_fn_kwargs['clean_logits'] = clean_logits
    results['complete'] = {}
    with model.trace(clean):
        for fn_name, fn in metric_fn.items():
            results['complete'][fn_name] = fn(model, metric_fn_kwargs).save()
    
    ##########
    # Loop over thresholds to run ablation
    ##########

    # get metric on thresholded graph
    for i, threshold in enumerate([1e16] + thresholds):
        if i != 0:
            results[threshold] = {}

        mask = get_mask(circuit, threshold, threshold_on_nodes=node_ablation)
        if not node_ablation:
            mask = prune(mask)

        if i != 0:
            if get_graph_info:
                results[threshold]['n_nodes'] = get_n_nodes(mask)
                results[threshold]['n_edges'] = get_n_edges((mask, architectural_graph) if node_ablation else mask)
                results[threshold]['avg_deg'] = 2 * results[threshold]['n_edges'] / (results[threshold]['n_nodes'] if results[threshold]['n_nodes'] > 0 else 1)
                results[threshold]['density'] = get_density((mask, architectural_graph) if node_ablation else mask)

        # get dict metric_name -> metric_values

        run_graph_args = {
            'model': model,
            'name2mod': name2mod,
            'dictionaries': sae_dict,
            'clean': clean,
            'patch': patch,
            'metric_fn': metric_fn,
            'metric_fn_kwargs': metric_fn_kwargs,
            'ablation_fn': ablation_fn,
            'complement': False,
        }
        if node_ablation:
            run_graph_args['architectural_graph'] = architectural_graph
            run_graph_args['mask'] = mask
        else:
            run_graph_args['computational_graph'] = mask

        threshold_result = run_graph(**run_graph_args)

        if i == 0:
            results['empty'] = threshold_result
        else:
            results[threshold]['faithfulness'] = {
                k: v.value.mean().item() for k, v in threshold_result.items()
            }

            for k in threshold_result:
                g = threshold_result[k].value
                m = results['complete'][k].value
                e = results['empty'][k].value
                faith = (g-e)/(m-e)
                faith = torch.where((m-e).abs() < 1e-6, torch.zeros_like(faith), faith)
                results[threshold]['faithfulness']['faithfulness_' + k] = faith.mean().item()

            run_graph_args['complement'] = True
            complement_result = run_graph(**run_graph_args)
            results[threshold]['completeness'] = {
                k: v.value.mean().item() for k, v in complement_result.items()
            }

            for k in complement_result:
                g = complement_result[k].value
                m = results['complete'][k].value
                e = results['empty'][k].value
                faith = (g-e)/(m-e)
                faith = torch.where((m-e).abs() < 1e-6, torch.zeros_like(faith), faith)
                results[threshold]['completeness']['completeness_' + k] = faith.mean().item()

    return results
