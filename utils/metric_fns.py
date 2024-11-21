import torch

from nnsight.models.UnifiedTransformer import UnifiedTransformer

# TODO : clean_logits should have the same size as logits

def metric_fn_logit(model, kwargs={}):
    """
    return the target logit
    requires trg_idx and at least one of trg_pos or trg_neg
    """
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    trg_idx = kwargs.get('trg_idx', None)
    trg_pos = kwargs.get('trg_pos', None)
    trg_neg = kwargs.get('trg_neg', None)

    if trg_pos is None and trg_neg is None:
        raise ValueError("Wrong arguments in metric_fn_logit")

    logits = module.output[torch.arange(trg_idx.numel()), trg_idx]
    res = 0
    if trg_pos is not None:
        if isinstance(trg_pos, torch.Tensor):
            res += logits[trg_pos]
        elif isinstance(trg_pos, list):
            for i, tokens in enumerate(trg_pos):
                res += logits[i, tokens].sum()
    if trg_neg is not None:
        if isinstance(trg_neg, torch.Tensor):
            res -= logits[trg_neg]
        elif isinstance(trg_neg, list):
            for i, tokens in enumerate(trg_neg):
                res -= logits[i, tokens].sum()
    return res

def metric_fn_logit_extended(model, kwargs={}):
    """
    Same as metric_fn_logit but do not aggregate over the batch dimension
    """
    """
    return the target logit
    requires trg_idx and at least one of trg_pos or trg_neg
    """
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    trg_idx = kwargs.get('trg_idx', None)
    trg_pos = kwargs.get('trg_pos', None)
    trg_neg = kwargs.get('trg_neg', None)

    if trg_pos is None and trg_neg is None:
        raise ValueError("Wrong arguments in metric_fn_logit")

    b = trg_idx.numel()
    logits = module.output[torch.arange(b), trg_idx] # (b, s, d_model) -> (b, d_model)
    res = torch.zeros_like(logits)[:, 0] # (b, d_model) -> (b), just to create a proxy tensor when logits is. If this still creates a tensor, clone logits, multiply by 0 and take the first column
    if trg_pos is not None:
        if isinstance(trg_pos, torch.Tensor):
            raise NotImplementedError("This case is not implemented. Check that trg_pos and logits are as expected")
            res = logits[torch.arange(b), trg_pos]
        elif isinstance(trg_pos, list):
            for i, tokens in enumerate(trg_pos):
                res[i] = logits[i, tokens].sum()
    if trg_neg is not None:
        if isinstance(trg_neg, torch.Tensor):
            raise NotImplementedError("This case is not implemented. Check that trg_neg and logits are as expected")
            res -= logits[torch.arange(b), trg_neg]
        elif isinstance(trg_neg, list):
            for i, tokens in enumerate(trg_neg):
                res[i] -= logits[i, tokens].sum()
    return res


def metric_fn_KL(model, kwargs={}):
    """
    return the KL divergence between the current logits and a target clean logits
    requires trg_idx and clean_logits
    """
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out

    clean_logits = kwargs.get('clean_logits', None)
    trg_idx = kwargs.get('trg_idx', None)
    if trg_idx is None or clean_logits is None:
        raise ValueError("Wrong arguments in metric_fn_KL")
    
    batch_size = trg_idx.numel()

    logits = module.output[torch.arange(batch_size), trg_idx] # (b, s, d_model) -> (b, d_model)
    clean_logits = clean_logits[torch.arange(batch_size), trg_idx]
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits, dim=-1),
        torch.nn.functional.log_softmax(clean_logits, dim=-1),
        reduction='none',
        log_target=True
    ).sum(dim=-1)

def metric_fn_statistical_distance(model, kwargs={}):
    """
    return the statistical distance between the current logits and a target clean logits
    """
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    
    trg_idx = kwargs.get('trg_idx', None)
    clean_logits = kwargs.get('clean_logits', None)
    if trg_idx is None or clean_logits is None:
        raise ValueError("Wrong arguments in metric_fn_statistical_distance")
    
    batch_size = trg_idx.numel()
    logits = module.output[torch.arange(batch_size), trg_idx] # (b, s, d_model) -> (b, d_model)
    clean_logits = clean_logits[torch.arange(batch_size), trg_idx]
    sigma = torch.nn.functional.softmax
    return (0.5 * torch.abs(sigma(logits, dim=-1) - sigma(clean_logits, dim=-1))).sum(dim=-1)

def metric_fn_acc(model, trg=None, clean_logits=None):
    """
    return 1 if the model's prediction is correct, 0 otherwise
    """
    raise NotImplementedError("This fails for some reason, nnsight don't want to do the last part implicitely")
    if trg is None:
        raise ValueError("trg must be provided")
    if clean_logits is None:
        raise ValueError("clean_logits must be provided")
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    batch_size = trg[0].numel()
    logits = module.output[torch.arange(batch_size), trg[0]] # (b, s, d_model) -> (b, d_model)
    clean_logits = clean_logits[torch.arange(batch_size), trg[0]]
    if isinstance(trg[1], torch.Tensor):
        g = (logits.argmax(dim=-1) == trg[1]).float()
        m = (clean_logits.argmax(dim=-1) == trg[1]).float()
        return 1 - ((g + m) - 2 * (g * m))
    elif isinstance(trg[1], list):
        res = []
        for i, tokens in enumerate(trg[1]):
            g = -1
            m = -1
            for token in tokens:
                if g == -1:
                    g = (logits[i].argmax().item() == token)
                    m = (clean_logits[i].argmax().item() == token)
                else:
                    g = g or (logits[i].argmax().item() == token)
                    m = m or (clean_logits[i].argmax().item() == token)
            g = g.item()
            m = m.item()
            res.append( 1 - ((g + m) - 2 * (g * m)) )
        return torch.tensor(res).float()

def metric_fn_MRR(model, trg=None, clean_logits=None):
    """
    default : return 1/rank of the correct answer
    """
    raise NotImplementedError("This fails for some reason, nnsight don't want to do the last part implicitely")
    if trg is None:
        raise ValueError("trg must be provided")
    if clean_logits is None:
        raise ValueError("clean_logits must be provided")
    if isinstance(model, UnifiedTransformer):
        module = model.unembed
    else:
        module = model.embed_out
    batch_size = trg[0].numel()
    logits = module.output[torch.arange(batch_size), trg[0]]
    clean_logits = clean_logits[torch.arange(batch_size), trg[0]]
    if isinstance(trg[1], torch.Tensor):
        g = (logits.argsort(dim=-1, descending=True) == trg[1].unsqueeze(-1)).float().argmax(dim=-1).float()
        m = (clean_logits.argsort(dim=-1, descending=True) == trg[1].unsqueeze(-1)).float().argmax(dim=-1).float()
        return (m+1)/(g+1)
    else:
        res = []
        for i, tokens in enumerate(trg[1]):
            g = -1
            m = -1
            for token in tokens:
                if g == -1:
                    g = (logits[i].argsort(descending=True) == token).float().argmax().float()
                    m = (clean_logits[i].argsort(descending=True) == token).float().argmax().float()
                else:
                    g = min(g, (logits[i].argsort(descending=True) == token).float().argmax().float())
                    m = min(m, (clean_logits[i].argsort(descending=True) == token).float().argmax().float())
            res.append( (m + 1) / (g + 1) )
        return torch.tensor(res)
