import torch

from ablation.edge_ablation import run_graph as run_edge_ablation
from ablation.node_ablation import run_graph as run_node_ablation

from utils.activation import SparseAct
from utils.ablation_fns import mean_ablation, zero_ablation, id_ablation
from utils.graph_utils import get_mask, prune, get_n_nodes, get_n_edges, get_density

# TODO : instead of clean and patch, give the buffer, to not re threshold for each batch.
# TODO : separate this function into one that computes metrics given a mask graph, and
#        one that gather all evaluations, not only faithfulness

@torch.no_grad()
def faithfulness(
        model,
        name2mod,
        sae_dict,
        clean,
        circuit,
        architectural_graph,
        thresholds,
        metric_fn,
        metric_fn_kwargs={},
        patch=None,
        ablation_fn=None,
        default_ablation='mean',
        get_graph_info=True,
        node_ablation=False
    ):
    """
    model : nnsight model
    submodules : list of model submodules
    sae_dict : dict
        dict [submodule] -> SAE
    name_dict : dict
        dict [submodule] -> str
    clean : str, list of str or tensor (batch, seq_len)
        the input to the model
    patch : None, str, list of str or tensor (batch, seq_len)
        the counterfactual input to the model to ablate edges.
        If None, ablation_fn is default to mean
        Else, ablation_fn is default to identity
    circuit : edges
    thresholds : float or list of float
        the thresholds to discard edges based on their weights
    metric_fn : callable or dict name -> callable
        the function(s) to evaluate the model.
        It can be CE, accuracy, logit for target token, etc.
    metric_fn_kwargs : dict
        the kwargs to pass to metric_fn. E.g. target token.
    ablation_fn : callable
        the function used to get the patched states.
        E.g. : mean ablation means across batch and sequence length or only across one of them.
               zero ablation : patch states are just zeros
               id ablation : patch states are computed from the patch input and left unchanged
    default_ablation : str
        the default ablation function to use if patch is None
        Available : mean, zero
        Mean is across batch and sequence length by default.

    returns a dict
        threshold -> dict ('TODO:nedges/nnodes/avgdegre/anyothermetriconthegraph', 'metric', 'metric_comp', 'faithfulness', 'completeness')
            -> float
        'complete' -> float (metric on the original model)
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

    # get metric on fully ablated model
    results['empty'] = {}
    if patch is None: patch = clean

    with model.trace(patch):
        x = name2mod['y'].module.output
        if type(x.shape) == tuple:
            x = x[0]
        x_hat, f = sae_dict['y'](x, output_features=True)
        last_state = SparseAct(act=f, res=x - x_hat).save()
    last_state = ablation_fn(last_state)
    with model.trace(patch):
        submodule = name2mod['y']
        if isinstance(submodule.module.output, tuple):
            submodule.module.output[0] = sae_dict['y'].decode(last_state.act) + last_state.res
        else:
            submodule.module.output = sae_dict['y'].decode(last_state.act) + last_state.res
        
        for fn_name, fn in metric_fn.items():
            results['empty'][fn_name] = fn(model, metric_fn_kwargs).save()
    
    ##########
    # Loop over thresholds to run ablation
    ##########

    # get metric on thresholded graph
    for i, threshold in enumerate(thresholds):
        results[threshold] = {}

        mask = get_mask(circuit, threshold, threshold_on_nodes=node_ablation)
        if not node_ablation:
            mask = prune(mask)

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
