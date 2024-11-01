from collections import namedtuple
import torch as t
from fancy_einsum import einsum
import einops
from tqdm import tqdm
from numpy import ndindex
from typing import Dict, Union
from utils.activation import SparseAct

DEBUGGING = False

if DEBUGGING:
    tracer_kwargs = {'validate' : True, 'scan' : True}
else:
    tracer_kwargs = {'validate' : False, 'scan' : False}

@t.no_grad()
def get_feature_effect(
    model,
    clean, hidden_states_clean, hidden_states_patch,
    dictionaries,
    downstream, # str : downstream module
    upstreams, # list of str : modules upstream of downstream
    name2mod, # dict str -> Submodule : name to module
    features_by_submod,
    is_tuple, steps,
    edge_threshold,
    metric_fn, metric_kwargs=dict(),
):
    """
    Get the effect of some upstream module on some downstream module. Uses integrated gradient attribution.
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
    
    effect_indices = {}
    effect_values = {}
    clean_states = [hidden_states_clean[upstream_submod] for upstream_submod in upstreams]
    patch_states = [hidden_states_patch[upstream_submod] for upstream_submod in upstreams]

    effect_indices = {}
    effect_values = {}
    for downstream_feat in downstream_features:
        metrics = []
        fs = {}

        for step in range(1, steps+1):
            alpha = step / steps

            reconstructed_input = 0

            for upstream_name in upstreams:
                upstream_act = (1 - alpha) * clean_states[upstream_name] + alpha * patch_states[upstream_name] # act shape (batch_size, seq_len, n_features) res shape (batch_size, seq_len, d_model)
                upstream_act.act = upstream_act.act.clone().detach().requires_grad_(True).retain_grad()
                upstream_act.res = upstream_act.res.clone().detach().requires_grad_(True).retain_grad()

                if fs.get(upstream_name) is None:
                    fs[upstream_name] = []
                fs[upstream_name].append(upstream_act)
                
                reconstructed_input += dictionaries[upstream_name].decode(upstream_act.act) + upstream_act.res
            
            if downstream != 'y':
                # TODO raise NotImplementedError("Check how to deal with inputs for all different modules !")
                # get the new output of the downstream module.
                if downstream_submod.LN is not None:
                    ln = downstream_submod.LN.forward(reconstructed_input)
                    y = downstream_submod.module.forward(ln)
                else:
                    y = downstream_submod.module.forward(reconstructed_input)
                
                y_hat, g = dictionaries[downstream](y, output_features=True)
                y_res = y - y_hat
                downstream_act = SparseAct(
                    act=g,
                    res=y_res
                )
                
                # Now, modify current feature in the downstream module's output and compute the effect on the metric
                if downstream_feat == downstream_act.act.size(-1):
                    diff = downstream_act.res - clean_states[downstream_submod].res
                else:
                    # Get the scalar value (coeff) of the downstream feature
                    diff = downstream_act.act[..., downstream_feat] - clean_states[downstream_submod].act[..., downstream_feat]
                    # Use this to scale the target feature vector
                    diff = diff * dictionaries[downstream].decoder.weight[:, downstream_feat]
            
                with model.trace(clean, **tracer_kwargs):
                    if is_tuple[downstream]:
                        downstream_submod.output[0] += diff # downstream_act - clean_states Remove the current feature's effect and add it's patched effect
                    else:
                        downstream_submod.output += diff

                    metrics.append(metric_fn(model, metric_kwargs))
            
            else:
                with model.trace(clean, **tracer_kwargs):
                    if is_tuple[downstream]:
                        downstream_submod.output[0] = reconstructed_input
                    else:
                        downstream_submod.output = reconstructed_input

                    metrics.append(metric_fn(model, metric_kwargs))


        metric = sum([m for m in metrics])
        metric.sum().backward(retain_graph=True)

        for upstream_name in upstreams:
            mean_act_grad = sum([f.act.grad for f in fs[upstream_name]]) / steps
            mean_res_grad = sum([f.res.grad for f in fs[upstream_name]]) / steps

            grad = SparseAct(act=mean_act_grad, res=mean_res_grad)
            delta = (patch_states[upstream_name] - clean_states[upstream_name]).detach()

            if effect_indices.get(upstream_name) is None:
                effect_indices[upstream_name] = {}
                effect_values[upstream_name] = {}

            effect = (grad @ delta).abs()

            effect_indices[upstream_name][downstream_feat] = t.where(
                effect.abs() > edge_threshold,
                effect,
                t.zeros_like(effect)
            ).nonzero().squeeze(-1)

            effect_values[upstream_name][downstream_feat] = effect[effect_indices[upstream_name][downstream_feat]]

    # get shapes for the return sparse tensors
    d_downstream_contracted = t.tensor(hidden_states_clean[downstream].act.size() if downstream != 'y' else (0,)) # if downstream == 'y', then the output is a single scalar
    d_downstream_contracted[-1] += 1
    d_downstream_contracted = d_downstream_contracted.prod()

    d_upstream_contracted = {}
    for upstream_name in upstreams:
        d_upstream_contracted[upstream_name] = t.tensor(clean_states[upstream_name].act.size())
        d_upstream_contracted[upstream_name][-1] += 1
        d_upstream_contracted[upstream_name] = d_upstream_contracted[upstream_name].prod

    # Create the sparse_coo_tensor containing the effect of each upstream module on the downstream module
    effects = {}
    for upstream_name in upstreams:
        # converts the dictionary of indices to a tensor of indices
        effect_indices[upstream_name] = t.tensor(
            [[downstream_feat for downstream_feat in downstream_features for _ in effect_indices[upstream_name][downstream_feat]],
            t.cat([effect_indices[upstream_name][downstream_feat] for downstream_feat in downstream_features], dim=0)]
        ).to(device).long()
        effect_values[upstream_name] = t.cat([effect_values[upstream_name][downstream_feat] for downstream_feat in downstream_features], dim=0)

        potential_upstream_features = effect_indices[upstream_name][1] # list of indices
        
        # TODO : use upstream_mask like in __old_get_effect :
        # upstream_mask = nodes[upstream_name].to_tensor().flatten()[potential_upstream_features].abs() > node_threshold
        if features_by_submod.get(upstream_name) is None:
            features_by_submod[upstream_name] = potential_upstream_features.unique().tolist()
        else:
            features_by_submod[upstream_name] += potential_upstream_features.unique().tolist()
            features_by_submod[upstream_name] = list(set(features_by_submod[upstream_name]))

        effects[upstream_name] = t.sparse_coo_tensor(
            effect_indices[upstream_name], effect_values[upstream_name],
            (d_downstream_contracted, d_upstream_contracted[upstream_name])
        )

    return effects

##########
# Old functions, not really used anymore
##########

EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])

def y_effect(
    model,
    clean,
    hidden_states_clean,
    hidden_states_patch,
    last_layer,
    submods,
    dictionaries,
    is_tuple,
    steps,
    metric_fn,
    metric_kwargs=dict(),
    normalise_edges=False,
    node_threshold=0.1,
    edge_threshold=0.01,
    features_by_submod={},
):
    """
    Get the last layer Integrated Gradient attribution effect on the output.
    """
    node_effects = {}
    for submod in submods:
        dictionary = dictionaries[submod]
        clean_state = hidden_states_clean[submod]
        patch_state = hidden_states_patch[submod]

        with model.trace(**tracer_kwargs) as tracer:
            metrics = []
            fs = []
            for step in range(1, steps+1):
                alpha = step / steps
                upstream_act = (1 - alpha) * clean_state + alpha * patch_state
                upstream_act.act.retain_grad()
                upstream_act.res.retain_grad()
                fs.append(upstream_act)
                with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                    if is_tuple[submod]:
                        submod.output[0][:] = dictionary.decode(upstream_act.act) + upstream_act.res
                    else:
                        submod.output = dictionary.decode(upstream_act.act) + upstream_act.res
                    metrics.append(metric_fn(model, metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward(retain_graph=True)

        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = (grad @ delta).abs()
        node_effects[submod] = effect

        if submod == last_layer:
            effect = effect.to_tensor().flatten()
            if normalise_edges:
                if clean_state.act.size(0) > 1:
                    raise NotImplementedError("Batch size > 1 not implemented yet.")
                tot_eff = effect.sum()
                effect = effect / tot_eff

                perm = t.argsort(effect, descending=True)
                perm_inv = t.argsort(perm)

                cumsum = t.cumsum(effect[perm], dim=0)
                mask = cumsum < edge_threshold
                first_zero_idx = t.where(mask == 0)[0][0]
                mask[first_zero_idx] = 1
                mask = mask[perm_inv]

                effect = t.where(
                    mask,
                    effect,
                    t.zeros_like(effect)
                ).to_sparse()
            else:
                effect = t.where(
                        effect.abs() > max(edge_threshold, node_threshold),
                        effect,
                        t.zeros_like(effect)
                    ).to_sparse()
                
            last_effect = effect
            features_by_submod[last_layer] = effect.coalesce().indices()[0].unique().tolist()

    return last_effect, node_effects

@t.no_grad()
def __old_get_effect(
    model,
    clean,
    hidden_states_clean,
    hidden_states_patch,
    dictionaries,
    layer,
    upstream_submod,
    downstream_submod,
    features_by_submod,
    is_tuple,
    steps,
    normalise_edges,
    nodes,
    node_threshold,
    edge_threshold,
):
    """
    Get the effect of some upstream module on some downstream module. Uses integrated gradient attribution.

    If normalise edges, divide them by their sum and take the smallest top k edges such that
    their sum is above edge_threshold * total sum or the smallest is equal to edge_threshold
    of the first one, to avoid cases where there is a shit ton of edges so the total is really big
    """
    try:
        downstream_features = features_by_submod[downstream_submod]
    except KeyError:
        raise ValueError(f"Module {downstream_submod} has no features to compute effects for")

    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'cfg'):
        device = model.cfg.device
    else:
        raise ValueError("Can't get model device :c")

    #print(f"Computing effects for layer {layer} with {len(downstream_features)} features")

    if not features_by_submod[downstream_submod]: # handle empty list
        raise ValueError(f"Module {downstream_submod} has no features to compute effects for")
    
    else:
        downstream_dict = dictionaries[downstream_submod]
        
        effect_indices = {}
        effect_values = {}
        dictionary = dictionaries[upstream_submod]
        clean_state = hidden_states_clean[upstream_submod]
        patch_state = hidden_states_patch[upstream_submod]
        
        Jack = 0

        for step in range(1, steps+1):
            alpha = step / steps
            upstream_act = (1 - alpha) * clean_state + alpha * patch_state # act shape (batch_size, seq_len, n_features) res shape (batch_size, seq_len, d_model)

            n_features = upstream_act.act.size(-1)
            
            def __jacobian_forward(act_res):
                act = act_res[..., :n_features]
                res = act_res[..., n_features:]
                with model.trace(clean, **tracer_kwargs):
                    if is_tuple[upstream_submod]:
                        upstream_submod.output[0][:] = dictionary.decode(act) + res
                    else:
                        upstream_submod.output = dictionary.decode(act) + res
                    
                    y = downstream_submod.output
                    if is_tuple[downstream_submod]:
                        y = y[0]
                    y_hat, g = downstream_dict(y, output_features=True)
                    y_res = y - y_hat
                    # the left @ down in the original code was at least useful to populate .resc instead of .res. I should just du .resc = norm of .res :
                    # all values represent the norm of their respective feature, so if we consider .res as a feature, then we should indeed
                    # consider its norm as the node.
                    # /!\ do the .to_tensor().flatten() outside of the with in order for the proxies to be populated and .to_tensor() to not crash
                    downstream_act = SparseAct(
                        act=g,
                        resc=t.norm(y_res, dim=-1)
                    ).save()
                
                downstream_act = downstream_act.to_tensor().flatten()
                downstream_act = downstream_act[downstream_features]
                return downstream_act
            
            # Jack shape : outshape x inshape = (# downstream features, batch_size, seq_len, n_features)
            Jack = t.autograd.functional.jacobian(__jacobian_forward, t.cat([upstream_act.act, upstream_act.res], dim=-1)) + Jack

        # get shapes
        d_downstream_contracted = t.tensor(hidden_states_clean[downstream_submod].act.size())
        d_downstream_contracted[-1] += 1
        d_downstream_contracted = d_downstream_contracted.prod()
        
        d_upstream_contracted = t.tensor(upstream_act.act.size())
        d_upstream_contracted[-1] += 1
        d_upstream_contracted = d_upstream_contracted.prod()

        # TODO : try to compile that loop
        for downstream_feat, fs_grad in zip(downstream_features, Jack):
            # fs_grad has shape (batch_size, seq_len, n_features + d_model) as it was concatenated for the jacobian
            # split it in a SparseAct{act : (batch_size, seq_len, n_features), res : (batch_size, seq_len, d_model)}
            fs_grad = SparseAct(
                act=fs_grad[..., :n_features],
                res=fs_grad[..., n_features:]
            )
            grad = fs_grad / steps
            delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
            effect = (grad @ delta).abs()
            
            effect = effect.to_tensor().flatten()

            if normalise_edges:
                """
                get non zero indices, get tot effect, divide by tot effect, get top edge_threshold * 100 % edges
                TOD?: second case (edge_threshold * max edge) if degree is too high
                """
                if clean_state.act.size(0) > 1:
                    raise NotImplementedError("Batch size > 1 not implemented yet.")

                effect_indices[downstream_feat] = effect.nonzero().squeeze(-1)
                effect_values[downstream_feat] = effect[effect_indices[downstream_feat]]
                tot_eff = effect_values[downstream_feat].sum()
                effect_values[downstream_feat] = effect_values[downstream_feat] / tot_eff

                perm = t.argsort(effect_values[downstream_feat], descending=True)
                cumsum = t.cumsum(effect_values[downstream_feat][perm], dim=0) # start at 0, end at 1

                mask = cumsum < edge_threshold # only ones then only zeros
                first_zero_idx = t.where(mask == 0)[0][0]
                mask[first_zero_idx] = 1
                effect_indices[downstream_feat] = effect_indices[downstream_feat][perm][mask]
                effect_values[downstream_feat] = effect_values[downstream_feat][perm][mask]

            else :
                effect_indices[downstream_feat] = t.where(
                    effect.abs() > edge_threshold,
                    effect,
                    t.zeros_like(effect)
                ).nonzero().squeeze(-1)

                effect_values[downstream_feat] = effect[effect_indices[downstream_feat]]

        # converts the dictionary of indices to a tensor of indices
        effect_indices = t.tensor(
            [[downstream_feat for downstream_feat in downstream_features for _ in effect_indices[downstream_feat]],
            t.cat([effect_indices[downstream_feat] for downstream_feat in downstream_features], dim=0)]
        ).to(device).long()
        effect_values = t.cat([effect_values[downstream_feat] for downstream_feat in downstream_features], dim=0)

        potential_upstream_features = effect_indices[1] # list of indices
        # now, in nodes[upstream_submod], we have the effect of upstream_submod on the final output. Keep only the features that have a total effect above the threshold
        upstream_mask = nodes[upstream_submod].to_tensor().flatten()[potential_upstream_features].abs() > node_threshold
        features_by_submod[upstream_submod] = potential_upstream_features[upstream_mask].unique().tolist()

        effect_indices = effect_indices[:, upstream_mask]
        effect_values = effect_values[upstream_mask]

        #print(f"Done computing effects for layer {layer}, found {len(effect_values)} edges & {len(features_by_submod[upstream_submod])} features")
        
        return t.sparse_coo_tensor(
            effect_indices, effect_values,
            (d_downstream_contracted, d_upstream_contracted)
        )

def head_attribution(
    clean,
    patch,
    model,
    attns,
    other_submods,
    dictionaries,
    metric_fn,
    steps=10,
    metric_kwargs=dict(),
    ablation_fn=lambda x : t.zeros_like(x) if isinstance(x, t.Tensor) else x.zeros_like(),
):
    submodules = attns + other_submods
    
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_"):
        for submodule in submodules:
            if submodule in attns:
                is_tuple[submodule.hook_z] = type(submodule.hook_z.output.shape) == tuple
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    def _get_hidden_states(hidden_states):
        i = 0
        for submodule in submodules:
            if submodule in attns:
                submodule = submodule.hook_z
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states[submodule] = SparseAct(act=f.save(), res=residual.save())

    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.no_grad():
        _get_hidden_states(hidden_states_clean)
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        patch = clean

    hidden_states_patch = {}
    with model.trace(patch, **tracer_kwargs), t.no_grad():
        _get_hidden_states(hidden_states_patch)
    hidden_states_patch = {k : ablation_fn(v.value) for k, v in hidden_states_patch.items()}

    effects = {}

    # First, deal with the attention heads
    for attn in attns:
        hook_z = attn.hook_z
        effects[hook_z] = []

        dictionary = dictionaries[hook_z]
        clean_state = hidden_states_clean[hook_z] # (batch_size, seq_len, n_head, d_head)
        patch_state = hidden_states_patch[hook_z] # (batch_size, seq_len, n_head, d_head)
        b, s, n_head, d_head = clean_state.act.shape
        for h in range(n_head):
            clean_kwargs = {}
            patch_kwargs = {}
            for attr in ['act', 'res']:
                if getattr(clean_state, attr) is not None:
                    clean_kwargs[attr] = getattr(clean_state, attr)[:, :, h, :] # (batch_size, seq_len, d_head)
                    patch_kwargs[attr] = getattr(patch_state, attr)[:, :, h, :] # (batch_size, seq_len, d_head)
            clean_state_h = SparseAct(**clean_kwargs)
            patch_state_h = SparseAct(**patch_kwargs)

            with model.trace(**tracer_kwargs) as tracer:
                metrics = []
                fs = []
                for step in range(1, steps+1):
                    alpha = step / steps
                    f_h = (1 - alpha) * clean_state_h + alpha * patch_state_h
                    f_h.act.retain_grad()
                    f_h.res.retain_grad()
                    fs.append(f_h)
                    f = SparseAct(clean_state.act.clone(), clean_state.res.clone())
                    f.act[:, :, h, :] = f_h.act
                    f.res[:, :, h, :] = f_h.res
                    # TODO : check that this gives the right grad on f_h
                    with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                        z = dictionary.decode(f.act) + f.res
                        attn_out = einsum(
                            "batch pos head_index d_head, \
                                head_index d_head d_model -> \
                                batch pos head_index d_model",
                            z,
                            attn.W_O,
                        )
                        attn_out = (
                            einops.reduce(attn_out, "batch position index model->batch position model", "sum")
                            + attn.b_O
                        )
                        if is_tuple[attn]:
                            attn.output[0][:] = attn_out
                        else:
                            attn.output = attn_out
                        metrics.append(metric_fn(model, **metric_kwargs))
                metric = sum([m for m in metrics])
                metric.sum().backward(retain_graph=True) # TODO : why retain_graph ?
            
            mean_grad = sum([f.act.grad for f in fs]) / steps
            mean_residual_grad = sum([f.res.grad for f in fs]) / steps
            grad = SparseAct(act=mean_grad, res=mean_residual_grad)
            delta = (patch_state_h - clean_state_h).detach() if patch_state is not None else -clean_state_h.detach()
            effect = grad @ delta
            effect = effect.act.sum(dim=-1) # sum over the d_head dimension

            effects[hook_z].append(effect)
        effects[hook_z] = t.stack(effects[hook_z], dim=-1) # from list of (batch_size, seq_len) to (batch_size, seq_len, n_head)
    
    # Now, the other submodules
    for submodule in other_submods:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule] # (batch_size, seq_len, d_model)
        patch_state = hidden_states_patch[submodule] # (batch_size, seq_len, d_model)
        with model.trace(**tracer_kwargs) as tracer:
            metrics = []
            fs = []
            for step in range(1, steps+1):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.retain_grad()
                f.res.retain_grad()
                fs.append(f)
                with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                    if is_tuple[submodule]:
                        submodule.output[0][:] = dictionary.decode(f.act) + f.res
                    else:
                        submodule.output = dictionary.decode(f.act) + f.res
                    metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward(retain_graph=True)
            
        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta
        effect = effect.act.sum(dim=-1)

        effects[submodule] = effect.unsqueeze(-1)

    return effects

##########
# Marks et al. functions. Here for compatibility.
##########

def _pe_attrib(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
):
    
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    hidden_states_clean = {}
    grads = {}
    with model.trace(clean, **tracer_kwargs):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True) # x_hat implicitly depends on f
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f, res=residual).save()
            grads[submodule] = hidden_states_clean[submodule].grad.save()
            residual.grad = t.zeros_like(residual)
            x_recon = x_hat + residual
            if is_tuple[submodule]:
                submodule.output[0][:] = x_recon
            else:
                submodule.output = x_recon
            x.grad = x_recon.grad
        metric_clean = metric_fn(model, **metric_kwargs).save()
        metric_clean.sum().backward()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}
    grads = {k : v.value for k, v in grads.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.inference_mode():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                x_hat, f = dictionary(x, output_features=True)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f, res=residual).save()
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        patch_state, clean_state, grad = hidden_states_patch[submodule], hidden_states_clean[submodule], grads[submodule]
        delta = patch_state - clean_state.detach() if patch_state is not None else -clean_state.detach()
        effect = delta @ grad
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    total_effect = total_effect if total_effect is not None else None
    
    return EffectOut(effects, deltas, grads, total_effect)

def _pe_ig(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        steps=10,
        metric_kwargs=dict(),
        ablation_fn=lambda x : t.zeros_like(x) if isinstance(x, t.Tensor) else x.zeros_like(),
):
    
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f.save(), res=residual.save())
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        patch=clean
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.no_grad():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f.save(), res=residual.save())
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : ablation_fn(v.value) for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        with model.trace(**tracer_kwargs) as tracer:
            metrics = []
            fs = []
            for step in range(1, steps+1):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.retain_grad()
                f.res.retain_grad()
                fs.append(f)
                with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                    if is_tuple[submodule]:
                        submodule.output[0][:] = dictionary.decode(f.act) + f.res
                    else:
                        submodule.output = dictionary.decode(f.act) + f.res
                    metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward(retain_graph=True)
            
        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return EffectOut(effects, deltas, grads, total_effect)

def _pe_exact(
    clean,
    patch,
    model,
    submodules,
    dictionaries,
    metric_fn,
    ):

    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.inference_mode():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f, res=residual).save()
        metric_clean = metric_fn(model).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.inference_mode():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f, res=residual).save()
            metric_patch = metric_fn(model).save()
        total_effect = metric_patch.value - metric_clean.value
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        effect = SparseAct(act=t.zeros_like(clean_state.act), resc=t.zeros(*clean_state.res.shape[:-1]))
        
        # iterate over positions and features for which clean and patch differ
        idxs = t.nonzero(patch_state.act - clean_state.act)
        for idx in tqdm(idxs):
            with t.inference_mode():
                with model.trace(clean, **tracer_kwargs):
                    f = clean_state.act.clone()
                    f[tuple(idx)] = patch_state.act[tuple(idx)]
                    x_hat = dictionary.decode(f)
                    if is_tuple[submodule]:
                        submodule.output[0][:] = x_hat + clean_state.res
                    else:
                        submodule.output = x_hat + clean_state.res
                    metric = metric_fn(model).save()
                effect.act[tuple(idx)] = (metric.value - metric_clean.value).sum()

        for idx in list(ndindex(effect.resc.shape)):
            with t.inference_mode():
                with model.trace(clean, **tracer_kwargs):
                    res = clean_state.res.clone()
                    res[tuple(idx)] = patch_state.res[tuple(idx)]
                    x_hat = dictionary.decode(clean_state.act)
                    if is_tuple[submodule]:
                        submodule.output[0][:] = x_hat + res
                    else:
                        submodule.output = x_hat + res
                    metric = metric_fn(model).save()
                effect.resc[tuple(idx)] = (metric.value - metric_clean.value).sum()
        
        effects[submodule] = effect
        deltas[submodule] = patch_state - clean_state
    total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, None, total_effect)

def patching_effect(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        method='attrib',
        steps=10,
        metric_kwargs=dict(),
        ablation_fn=None,
):
    if method == 'attrib':
        return _pe_attrib(clean, patch, model, submodules, dictionaries, metric_fn, metric_kwargs=metric_kwargs)
    elif method == 'ig':
        return _pe_ig(clean, patch, model, submodules, dictionaries, metric_fn, steps=steps, metric_kwargs=metric_kwargs, ablation_fn=ablation_fn)
    elif method == 'exact':
        return _pe_exact(clean, patch, model, submodules, dictionaries, metric_fn)
    else:
        raise ValueError(f"Unknown method {method}")

def jvp(
        input,
        model,
        dictionaries,
        downstream_submod,
        downstream_features,
        upstream_submod,
        left_vec : Union[SparseAct, Dict[int, SparseAct]],
        right_vec : SparseAct,
        return_without_right = False,
):
    """
    Return a sparse shape [# downstream features + 1, # upstream features + 1] tensor of Jacobian-vector products.
    """
    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'cfg'):
        device = model.cfg.device
    else:
        raise ValueError("Can't get model device :c")

    if not downstream_features: # handle empty list
        if not return_without_right:
            return t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(device)
        else:
            return t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(device), t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(device)

    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_"):
        is_tuple[upstream_submod] = type(upstream_submod.output.shape) == tuple
        is_tuple[downstream_submod] = type(downstream_submod.output.shape) == tuple

    downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]

    vjv_indices = {}
    vjv_values = {}
    if return_without_right:
        jv_indices = {}
        jv_values = {}

    with model.trace(input, **tracer_kwargs):
        # first specify forward pass modifications
        x = upstream_submod.output
        if is_tuple[upstream_submod]:
            x = x[0]
        x_hat, f = upstream_dict(x, output_features=True)
        x_res = x - x_hat
        upstream_act = SparseAct(act=f, res=x_res).save()
        if is_tuple[upstream_submod]:
            upstream_submod.output[0][:] = x_hat + x_res
        else:
            upstream_submod.output = x_hat + x_res
        y = downstream_submod.output
        if is_tuple[downstream_submod]:
            y = y[0]
        y_hat, g = downstream_dict(y, output_features=True)
        y_res = y - y_hat
        downstream_act = SparseAct(act=g, res=y_res).save()

        for downstream_feat in downstream_features:
            if isinstance(left_vec, SparseAct):
                to_backprop = (left_vec @ downstream_act).to_tensor().flatten()
            elif isinstance(left_vec, dict):
                to_backprop = (left_vec[downstream_feat] @ downstream_act).to_tensor().flatten()
            else:
                raise ValueError(f"Unknown type {type(left_vec)}")
            vjv = (upstream_act.grad @ right_vec).to_tensor().flatten()
            if return_without_right:
                jv = (upstream_act.grad @ right_vec).to_tensor().flatten()
            x_res.grad = t.zeros_like(x_res)
            to_backprop[downstream_feat].backward(retain_graph=True)

            vjv_indices[downstream_feat] = vjv.nonzero().squeeze(-1).save()
            vjv_values[downstream_feat] = vjv[vjv_indices[downstream_feat]].save()

            if return_without_right:
                jv_indices[downstream_feat] = jv.nonzero().squeeze(-1).save()
                jv_values[downstream_feat] = jv[vjv_indices[downstream_feat]].save()

    # get shapes
    d_downstream_contracted = len((downstream_act.value @ downstream_act.value).to_tensor().flatten())
    d_upstream_contracted = len((upstream_act.value @ upstream_act.value).to_tensor().flatten())
    if return_without_right:
        d_upstream = len(upstream_act.value.to_tensor().flatten())

    vjv_indices = t.tensor(
        [[downstream_feat for downstream_feat in downstream_features for _ in vjv_indices[downstream_feat].value],
         t.cat([vjv_indices[downstream_feat].value for downstream_feat in downstream_features], dim=0)]
    ).to(device)
    vjv_values = t.cat([vjv_values[downstream_feat].value for downstream_feat in downstream_features], dim=0)

    if not return_without_right:
        return t.sparse_coo_tensor(vjv_indices, vjv_values, (d_downstream_contracted, d_upstream_contracted))

    jv_indices = t.tensor(
        [[downstream_feat for downstream_feat in downstream_features for _ in jv_indices[downstream_feat].value],
         t.cat([jv_indices[downstream_feat].value for downstream_feat in downstream_features], dim=0)]
    ).to(device)
    jv_values = t.cat([jv_values[downstream_feat].value for downstream_feat in downstream_features], dim=0)

    return (
        t.sparse_coo_tensor(vjv_indices, vjv_values, (d_downstream_contracted, d_upstream_contracted)),
        t.sparse_coo_tensor(jv_indices, jv_values, (d_downstream_contracted, d_upstream))
    )