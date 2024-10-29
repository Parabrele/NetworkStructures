import graph_tool.all as gt

import time

import numpy as np
import torch
from welford_torch import OnlineCovariance

from tqdm import tqdm

from utils.activation import get_is_tuple, get_hidden_states, get_hidden_attr
from utils.metric_fns import metric_fn_logit
from utils.ablation_fns import zero_ablation
from data.buffer import unpack_batch

def generate_cov(
    data_buffer,
    model,
    embed,
    resids,
    attns,
    mlps,
    dictionaries,
    get_act=True,
    get_attr=True,
    n_batches=1000,
    metric_fn=metric_fn_logit,
    ablation_fn=zero_ablation,
    steps=10,
):
    """
    Get the covariance matrix of the activations of the model for some specified module and dictionary
    """
    submods = []
    if embed is not None:
        submods.append(embed)
    if resids is not None:
        submods += resids
    if attns is not None:
        submods += attns
    if mlps is not None:
        submods += mlps

    cov = {}
    if get_attr:
        cov['attr'] = {}
        for submod in submods:
            cov['attr'][submod] = OnlineCovariance()
    if get_act:
        cov['act'] = {}
        for submod in submods:
            cov['act'][submod] = OnlineCovariance()
    
    # get the shape of the hidden states
    is_tuple = get_is_tuple(model, submods)
    
    # there is an annoying bug with multiprocessing and nnsight. Don't use multiprocessing for now.
    i = 0
    for batch in tqdm(data_buffer, total=n_batches):
        tokens, trg_idx, trg, corr, corr_trg = unpack_batch(batch)
        
        if get_act:
            act = get_hidden_states(
                model, submods, dictionaries, is_tuple, tokens,
                reconstruction_error=True
            )
            for key in act:
                cov['act'][key].add_all(act[key].act)
                print("act.act.shape : ", act[key].act.shape)
        if get_attr:
            attr = get_hidden_attr(
                model, submods, dictionaries, is_tuple, tokens,
                metric_fn=metric_fn, patch=corr, ablation_fn=ablation_fn,
                steps=steps, metric_kwargs={"trg_idx": trg_idx, "trg_pos": trg, "trg_neg": corr_trg},
                reconstruction_error=True
            )
            for key in attr:
                cov['attr'][key].add_all(attr[key].act)
        i += 1
        if i == n_batches:
            break
        
        # with ProcessPoolExecutor(max_workers=available_gpus) as executor:
        #     futures = []
        #     for gpu in range(available_gpus):
        #         tokens, trg_idx, trg = next(buffer)
        #         args_dict_per_device[gpu]['clean'] = tokens.to(f'cuda:{gpu}')
        #         args_dict_per_device[gpu]['metric_fn_kwargs']['trg'] = (trg_idx.to(f'cuda:{gpu}'), trg.to(f'cuda:{gpu}'))
        #         futures.append(executor.submit(run_task, args_dict_per_device[gpu]))
        #     finished = False
        #     while not finished:
        #         for future in tqdm(as_completed(futures), total=None):
        #             # get the result
        #             act = future.result()
        #             print("Got act from future : ")
        #             for key in act:
        #                 print(key, act[key].shape)
        #                 current_gpu = act[key].device.index
        #                 cov[key].add_all(act[key].to(device))
        #             i += 1
        #             # remove the current future and add a new one
        #             futures.remove(future)
        #             if i == n_batches:
        #                 finished = True
        #                 break

        #             tokens, trg_idx, trg = next(buffer)
        #             args_dict_per_device[current_gpu]['clean'] = tokens.to(f'cuda:{current_gpu}')
        #             args_dict_per_device[current_gpu]['metric_fn_kwargs']['trg'] = (trg_idx.to(f'cuda:{current_gpu}'), trg.to(f'cuda:{current_gpu}'))
        #             futures.append(executor.submit(run_task, args_dict_per_device[current_gpu]))
            
        #     # kill all futures left
        #     for future in futures:
        #         future.cancel()

    return cov

def fit_sbm(cov):
    """
    cov : OnlineCovariance

    returns : torch.Tensor (block assignment for each feature), SBM state (graph_tool object)

    Fit a stochastic block model to the covariance between features.
    """

    # first remove dead features & define the weight matrix
    corr = cov.cov.cpu()
    eps = 1e-5
    diag = corr.diag().clone()
    dead_idx = diag < eps
    corr = corr[~dead_idx][:, ~dead_idx]
    diag = diag[~dead_idx]
    corr = corr / torch.sqrt(diag.unsqueeze(0) * diag.unsqueeze(1))
    diag_idx = torch.arange(corr.size(0))
    corr[diag_idx, diag_idx] = 0
    weights = 2 * torch.arctanh(corr)

    edge_list = torch.ones_like(weights).nonzero().cpu().numpy()
    weight_assignment = weights.numpy()[edge_list[:, 0], edge_list[:, 1]]

    # build the graph
    G_gt = gt.Graph(directed=False)
    G_gt.add_edge_list(edge_list)
    G_gt.ep['weight'] = G_gt.new_edge_property("float", vals=weight_assignment)

    # fit the SBM
    state_args = {
        'recs': [G_gt.ep['weight']],
        'rec_types': ['real-normal'],
    }

    print("Minimizing nested blockmodel...", end="")
    start = time.time()
    state = gt.minimize_nested_blockmodel_dl(G_gt, state_args=state_args)
    end = time.time()
    print(f"Done in {end - start} seconds.")
    print(f"Before refinement :\n{state.print_summary()}")

    # refine the SBM
    print("Refining nested blockmodel...", end="")
    start = time.time()
    for i in tqdm(range(100)):
        state.multiflip_mcmc_sweep(niter=10, beta=np.inf)
        # TODO : stop if converged
    end = time.time()
    print(f"Done in {end - start} seconds.")
    print(f"After refinement :\n{state.print_summary()}")

    # project the SBM to the relevant level
    # TODO : sometimes nested SBMs don't do much in the first(s) level(s), so we might want to project to the n-th.
    # How to choose n?
    # Among levels with less than 1/10 blocks/features,
    # first one that has more blocks than attn heads seems reasonable for comparison with heads.
    # But this might be a bit too small and more blocks might be better.
    raise NotImplementedError
    blockstate = state.project_level(1)#state.get_levels()[1]
    b = gt.contiguous_map(blockstate.get_blocks())
    blockstate = blockstate.copy(b=b)

    # get the block assignment for each feature, all dead are in one block
    propert_map = blockstate.get_blocks()
    propert_map = torch.tensor(propert_map)
    
    B = blockstate.get_nonempty_B()
    n_samples = train_dataset.data.size(0)

    roi_labels = torch.zeros(cov.cov.size(0)).long().to(device)
    roi_labels[dead_idx] = B
    roi_labels[~dead_idx] = propert_map.long().to(device)

    return roi_labels, state
