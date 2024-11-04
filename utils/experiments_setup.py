import torch

from nnsight import LanguageModel
from nnsight.models.UnifiedTransformer import UnifiedTransformer
from utils.utils import Submod
from utils.dictionary import IdentityDict, LinearDictionary, AutoEncoder

def load_model_and_modules(device, model_name="EleutherAI/pythia-70m-deduped", embed=True, resid=True, attn=True, mlp=True):
    model = UnifiedTransformer(
        model_name,
        device=device,
        processing=False,
    )
    model.device = model.cfg.device
    model.tokenizer.padding_side = 'left'

    name2mod = {
        'y' : Submod('y', model.blocks[-1])
    }
    if embed:
        name2mod['embed'] = Submod('embed', model.embed)
    
    for i in range(len(model.blocks)):
        if attn:
            name2mod[f'attn_{i}'] = Submod(f'attn_{i}', model.blocks[i].attn, model.blocks[i].ln1)
        if mlp:
            name2mod[f'mlp_{i}'] = Submod(f'mlp_{i}', model.blocks[i].mlp, model.blocks[i].ln2)
        if resid:
            name2mod[f'resid_{i}'] = Submod(f'resid_{i}', model.blocks[i])
        
    return model, name2mod

def get_architectural_graph(model, submods):
    """
    Returns a dict str -> list[str] of downstream -> upstream modules.
    """

    # Build the full graph, then remove nodes not in submods.
    graph = {
        'embed' : [],
        'attn_0' : ['embed'],
        'mlp_0' : ['embed'] if model.blocks[0].cfg.parallel_attn_mlp else ['embed', 'attn_0'],
        'resid_0' : ['attn_0', 'mlp_0', 'embed'],
        'y' : [f'resid_{len(model.blocks)-1}']
    }
    for i in range(1, len(model.blocks)):
        graph[f'attn_{i}'] = [f'resid_{i-1}']
        graph[f'mlp_{i}'] = [f'resid_{i-1}'] if model.blocks[i].cfg.parallel_attn_mlp else [f'resid_{i-1}', f'attn_{i}']
        graph[f'resid_{i}'] = [f'attn_{i}', f'mlp_{i}', f'resid_{i-1}']
    
    # Remove nodes not in submods
    for i in range(len(model.blocks)-1, -1, -1):
        if f'attn_{i}' not in submods:
            graph[f'resid_{i}'].remove(f'attn_{i}')
        if f'mlp_{i}' not in submods:
            graph[f'resid_{i}'].remove(f'mlp_{i}')
        if f'resid_{i}' not in submods:
            r = f'resid_{i}'
            for downstream in graph:
                if r in graph[downstream]:
                    graph[downstream].remove(r)
                    graph[downstream] += graph[r]
                    graph[downstream] = list(set(graph[downstream]))
            del graph[r]
    
    if 'embed' not in submods:
        for downstream in graph:
            graph[downstream].remove('embed')

    return graph

def load_saes(
    model,
    name2mod,
    idd=False,
    svd=False,
    white=False,
    device='cpu',
    path='/scratch/pyllm/dhimoila/',
    unified=True
):
    if white:
        raise NotImplementedError("Whitening is not implemented yet.")
    dictionaries = {}

    d_model = 512
    dict_size = 32768 if not svd else 512

    if idd:
        for name in name2mod:
            dictionaries[name] = IdentityDict(d_model)

        return dictionaries
    
    path = path + "dictionaires/pythia-70m-deduped/"# + ("SVDdicts/" if svd else "")

    if not svd:
        ae = AutoEncoder(d_model, dict_size).to(device)
        ae.load_state_dict(torch.load(path + f"embed/ae.pt", map_location=device))
        dictionaries['embed'] = ae
    else:
        d = torch.load(path + f"embed/cov.pt", map_location=device)
        mean = d['mean']
        cov = d['cov']
        U, S, V = torch.svd(cov)
        dictionaries['embed'] = LinearDictionary(d_model, dict_size)
        dictionaries['embed'].E = V.T
        dictionaries['embed'].D = V
        dictionaries['embed'].bias = mean

    for layer in range(len(model.gpt_neox.layers if not unified else model.blocks)):
        
        if not svd:
            ae = AutoEncoder(d_model, dict_size).to(device)
            ae.load_state_dict(torch.load(path + f"resid_out_layer{layer}/ae.pt", map_location=device))
            dictionaries[f'resid_{layer}'] = ae
        else:
            d = torch.load(path + f"resid_out_layer{layer}/cov.pt", map_location=device)
            mean = d['mean']
            cov = d['cov']
            U, S, V = torch.svd(cov) # cov is symmetric so U = V
            dictionaries[f'resid_{layer}'] = LinearDictionary(d_model, dict_size)
            dictionaries[f'resid_{layer}'].E = V.T
            dictionaries[f'resid_{layer}'].D = V
            dictionaries[f'resid_{layer}'].bias = mean
        
        if not svd:
            ae = AutoEncoder(d_model, dict_size).to(device)
            ae.load_state_dict(torch.load(path + f"attn_out_layer{layer}/ae.pt", map_location=device))
            dictionaries[f'attn_{layer}'] = ae
        else:
            d = torch.load(path + f"attn_out_layer{layer}/cov.pt", map_location=device)
            mean = d['mean']
            cov = d['cov']
            U, S, V = torch.svd(cov)
            dictionaries[f'attn_{layer}'] = LinearDictionary(d_model, dict_size)
            dictionaries[f'attn_{layer}'].E = V.T # This will perform the operation x @ E.T = x @ V, but V is in it's transposed form
            dictionaries[f'attn_{layer}'].D = V
            dictionaries[f'attn_{layer}'].bias = mean

        if not svd:
            ae = AutoEncoder(d_model, dict_size).to(device)
            ae.load_state_dict(torch.load(path + f"mlp_out_layer{layer}/ae.pt", map_location=device))
            dictionaries[f'mlp_{layer}'] = ae
        else:
            d = torch.load(path + f"mlp_out_layer{layer}/cov.pt", map_location=device)
            mean = d['mean']
            cov = d['cov']
            U, S, V = torch.svd(cov)
            dictionaries[f'mlp_{layer}'] = LinearDictionary(d_model, dict_size)
            dictionaries[f'mlp_{layer}'].E = V.T
            dictionaries[f'mlp_{layer}'].D = V
            dictionaries[f'mlp_{layer}'].bias = mean
    
    return dictionaries

# TODO : delete the below functions after testing the new pipeline

def __old_load_model_and_modules(device, unified=True, model_name="EleutherAI/pythia-70m-deduped"):
    if unified:
        model = UnifiedTransformer(
            model_name,
            device=device,
            processing=False,
        )
        model.device = model.cfg.device
        model.tokenizer.padding_side = 'left'

        embed = model.embed

        resids = []
        attns = []
        mlps = []
        for layer in range(len(model.blocks)):
            resids.append(model.blocks[layer])
            attns.append(model.blocks[layer].attn)
            mlps.append(model.blocks[layer].mlp)

        submod_names = {
            model.embed : 'embed'
        }
        for i in range(len(model.blocks)):
            submod_names[model.blocks[i].attn] = f'attn_{i}'
            submod_names[model.blocks[i].mlp] = f'mlp_{i}'
            submod_names[model.blocks[i]] = f'resid_{i}'

        return model, embed, resids, attns, mlps, submod_names
    
    else:
        if model_name != "EleutherAI/pythia-70m-deduped":
            raise NotImplementedError("Only EleutherAI/pythia-70m-deduped is supported for non-unified models.")
        model = LanguageModel(
            model_name,
            device_map=device,
            dispatch=True,
        )

        embed = model.gpt_neox.embed_in

        resids = []
        attns = []
        mlps = []
        for layer in range(len(model.gpt_neox.layers)):
            resids.append(model.gpt_neox.layers[layer])
            attns.append(model.gpt_neox.layers[layer].attention)
            mlps.append(model.gpt_neox.layers[layer].mlp)

        submod_names = {
            model.gpt_neox.embed_in : 'embed'
        }
        for i in range(len(model.gpt_neox.layers)):
            submod_names[model.gpt_neox.layers[i].attention] = f'attn_{i}'
            submod_names[model.gpt_neox.layers[i].mlp] = f'mlp_{i}'
            submod_names[model.gpt_neox.layers[i]] = f'resid_{i}'

        return model, embed, resids, attns, mlps, submod_names

def __old_load_saes(
    model,
    name2mod,
    idd=False,
    svd=False,
    white=False,
    device='cpu',
    path='/scratch/pyllm/dhimoila/',
    unified=True
):
    if white:
        raise NotImplementedError("Whitening is not implemented yet.")
    dictionaries = {}

    d_model = 512
    dict_size = 32768 if not svd else 512

    if idd:
        dictionaries[model_embed] = IdentityDict(d_model)
        for layer in range(len(model.gpt_neox.layers if not unified else model.blocks)):
            dictionaries[model_resids[layer]] = IdentityDict(d_model)
            dictionaries[model_attns[layer]] = IdentityDict(d_model)
            dictionaries[model_mlps[layer]] = IdentityDict(d_model)

        return dictionaries
    
    path = path + "dictionaires/pythia-70m-deduped/"# + ("SVDdicts/" if svd else "")

    if not svd:
        ae = AutoEncoder(d_model, dict_size).to(device)
        ae.load_state_dict(torch.load(path + f"embed/ae.pt", map_location=device))
        dictionaries[model_embed] = ae
    else:
        d = torch.load(path + f"embed/cov.pt", map_location=device)
        mean = d['mean']
        cov = d['cov']
        U, S, V = torch.svd(cov)
        dictionaries[model_embed] = LinearDictionary(d_model, dict_size)
        dictionaries[model_embed].E = V.T
        dictionaries[model_embed].D = V
        dictionaries[model_embed].bias = mean

    for layer in range(len(model.gpt_neox.layers if not unified else model.blocks)):
        
        if not svd:
            ae = AutoEncoder(d_model, dict_size).to(device)
            ae.load_state_dict(torch.load(path + f"resid_out_layer{layer}/ae.pt", map_location=device))
            dictionaries[model_resids[layer]] = ae
        else:
            d = torch.load(path + f"resid_out_layer{layer}/cov.pt", map_location=device)
            mean = d['mean']
            cov = d['cov']
            U, S, V = torch.svd(cov) # cov is symmetric so U = V
            dictionaries[model_resids[layer]] = LinearDictionary(d_model, dict_size)
            dictionaries[model_resids[layer]].E = V.T
            dictionaries[model_resids[layer]].D = V
            dictionaries[model_resids[layer]].bias = mean
        
        if not svd:
            ae = AutoEncoder(d_model, dict_size).to(device)
            ae.load_state_dict(torch.load(path + f"attn_out_layer{layer}/ae.pt", map_location=device))
            dictionaries[model_attns[layer]] = ae
        else:
            d = torch.load(path + f"attn_out_layer{layer}/cov.pt", map_location=device)
            mean = d['mean']
            cov = d['cov']
            U, S, V = torch.svd(cov)
            dictionaries[model_attns[layer]] = LinearDictionary(d_model, dict_size)
            dictionaries[model_attns[layer]].E = V.T # This will perform the operation x @ E.T = x @ V, but V is in it's transposed form
            dictionaries[model_attns[layer]].D = V
            dictionaries[model_attns[layer]].bias = mean

        if not svd:
            ae = AutoEncoder(d_model, dict_size).to(device)
            ae.load_state_dict(torch.load(path + f"mlp_out_layer{layer}/ae.pt", map_location=device))
            dictionaries[model_mlps[layer]] = ae
        else:
            d = torch.load(path + f"mlp_out_layer{layer}/cov.pt", map_location=device)
            mean = d['mean']
            cov = d['cov']
            U, S, V = torch.svd(cov)
            dictionaries[model_mlps[layer]] = LinearDictionary(d_model, dict_size)
            dictionaries[model_mlps[layer]].E = V.T
            dictionaries[model_mlps[layer]].D = V
            dictionaries[model_mlps[layer]].bias = mean
    
    return dictionaries
