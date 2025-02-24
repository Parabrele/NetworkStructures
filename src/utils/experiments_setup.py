import graph_tool.all as gt

import os

import random
import numpy as np

import torch

from nnsight.models.UnifiedTransformer import UnifiedTransformer

from sae_lens import SAE

from utils.utils import Submod
from utils.dictionary import IdentityDict

# TODO : load model with from_pretrained_no_processing(..., **cfg.model_from_pretrained_kwargs)"

def seed_all(seed: int):
    """
    Set random seed for all libraries.
    To be used only in sequential code.
    """
    # Set python hash seed to disable hash randomization and make hash based data structures / operations deterministic
    os.environ["PYTHONHASHSEED"] = "0"

    # Set random seed for all libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gt.seed_rng(seed)


model_name_to_processing = {
    "pythia-70m-deduped" : False,
    "gpt2" : True,
    "gemma-2-2b" : False,
}

model_name_to_sae_release = {
    "pythia-70m-deduped" : "pythia-70m-deduped-{M}-sm",
    "gpt2" : "gpt2-small-{M}-jb",
    "gemma-2-2b" : "gemma-scope-2b-pt-{M}-canonical",
}

model_hook_resid_pre = {
    "pythia-70m-deduped" : False,
    "gpt2" : True,
    "gemma-2-2b" : False,
}

DEFAULT_IDS = {
    "pythia-70m-deduped" : {
        "embed" : "blocks.0.hook_resid_pre",
        "resid" : "blocks.{L}.hook_resid_post",
        "attn" : "blocks.{L}.hook_attn_out",
        "mlp" : "blocks.{L}.hook_mlp_out",
    },
    "gpt2" : {
        "embed" : "blocks.0.hook_resid_pre",
        "resid" : "blocks.{L}.hook_resid_pre",
    },
    "gemma-2-2b" : {
        "resid" : "layer_{L}/width_65k/canonical",
        "attn" : "layer_{L}/width_65k/canonical",
        "mlp" : "layer_{L}/width_65k/canonical",
    }
}

def load_model_and_modules(device, model_name="pythia-70m-deduped", resid=True, attn=True, mlp=True, start_at_layer=-1, processing=None):
    """
    start_at_layer : int
        The layer to start loading modules from. If -1, load all layers. If 0, start from 'resid_0'.
    """
    processing = model_name_to_processing.get(model_name, None)
    if processing is None:
        raise ValueError(f"Model {model_name} has no default processing setting. Provide your own. Processing is used as argument for the UnifiedTransformer model and handles wrapping of LN and such into the rest of the weights.")
    model = UnifiedTransformer(
        model_name,
        device=device,
        processing=processing,
    )
    model.device = model.cfg.device
    model.tokenizer.padding_side = 'left'

    name2mod = {
        'y' : Submod('y', model.blocks[-1])
    }
    if start_at_layer == -1:
        name2mod['embed'] = Submod('embed', model.embed)
    
    for i in range(max(0, start_at_layer), len(model.blocks)):
        if attn and start_at_layer < i:
            name2mod[f'attn_{i}'] = Submod(f'attn_{i}', model.blocks[i].attn, model.blocks[i].ln1, model.blocks[i].ln1_post)
        if mlp and start_at_layer < i:
            name2mod[f'mlp_{i}'] = Submod(f'mlp_{i}', model.blocks[i].mlp, model.blocks[i].ln2, model.blocks[i].ln2_post)
        if resid or start_at_layer == i:
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
            del graph[f'attn_{i}']
        if f'mlp_{i}' not in submods:
            graph[f'resid_{i}'].remove(f'mlp_{i}')
            del graph[f'mlp_{i}']
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
            if 'embed' in graph[downstream]:
                graph[downstream].remove('embed')
        del graph['embed']

    return graph

def load_saes(
    model,
    name2mod,
    idd=False,
    release=None,
    id_dict=DEFAULT_IDS, # SAE-lens IDs for each sae. Should be a dict model_name -> module_type -> id.
    hook_resid_pre=None, # Whether SAEs were trained on resid pre or resid post.
):
    dictionaries = {}

    if idd:
        for name in name2mod:
            dictionaries[name] = IdentityDict()
        return dictionaries

    model_name = model.cfg.model_name
    if release is None:
        release = model_name_to_sae_release.get(model_name, None)
    if hook_resid_pre is None:
        hook_resid_pre = model_hook_resid_pre.get(model_name, None)
    if release is None:
        raise ValueError(f"Model {model_name} not supported by default. Provide your own release and id_dict.")
    if model_name not in id_dict:
        raise ValueError(f"Model {model_name} not supported by default. Provide your own id_dict.")
    if hook_resid_pre is None:
        raise ValueError(f"Model {model_name} not supported by default. Provide your own hook_resid_pre.")
    
    if 'embed' in name2mod:
        sae_id = id_dict[model_name].get('embed', None)
        if sae_id is None:
            raise ValueError(f"Current settings for model {model_name} does not support an SAE for the embedding layer. Provide your own in id_dict.")
        dictionaries['embed'] = SAE.from_pretrained(release.format(M='res'), sae_id)[0]

    for layer in range(len(model.blocks)):
        if f'resid_{layer}' in name2mod:
            sae_id = id_dict[model_name].get('resid', None)
            if sae_id is None:
                raise ValueError(f"Current settings for model {model_name} does not support an SAE for the residual layers. Provide your own in id_dict.")
            replacement_for_gpt2small_jb_why_do_you_not_do_as_everyone_else = "blocks.11.hook_resid_post"
            dictionaries[f'resid_{layer}'] = SAE.from_pretrained(
                release.format(M='res'),
                sae_id.format(L=(layer+1 if hook_resid_pre else layer)) if not (model_name == "gpt2" and layer == 11) else replacement_for_gpt2small_jb_why_do_you_not_do_as_everyone_else
            )[0]
        
        if f'attn_{layer}' in name2mod:
            sae_id = id_dict[model_name].get('attn', None)
            if sae_id is None:
                raise ValueError(f"Current settings for model {model_name} does not support an SAE for the attention layers. Provide your own in id_dict.")
            dictionaries[f'attn_{layer}'] = SAE.from_pretrained(
                release.format(M='att'),
                sae_id.format(L=layer)
            )[0]

        if f'mlp_{layer}' in name2mod:
            sae_id = id_dict[model_name].get('mlp', None)
            if sae_id is None:
                raise ValueError(f"Current settings for model {model_name} does not support an SAE for the MLP layers. Provide your own in id_dict.")
            dictionaries[f'mlp_{layer}'] = SAE.from_pretrained(
                release.format(M='mlp'),
                sae_id.format(L=layer)
            )[0]
    
    # TODO : check that 'y' SAE is never used.
    # try:
    #     TODO : load this one separately since some times "resid" modules might not be in name2mod.
    #     sae_id = id_dict[model_name].get('resid', None)
    #         if sae_id is None:
    #             raise ValueError(f"Current settings for model {model_name} does not support an SAE for the residual layers. Provide your own in id_dict.")
    #         dictionaries[f'resid_{layer}'] = SAE.from_pretrained(
    #             release.format(M='res'),
    #             sae_id.format(L=(layer+1 if hook_resid_pre else layer))
    #         )[0]
    #     dictionaries['y'] = dictionaries[f'resid_{len(model.blocks)-1}']
    # except KeyError:
    #     raise ValueError(f"Residual SAE for the final layer must be provided.")
    # TODO : remove that, just to check that it is never used.
    # for k, v in dictionaries.items():
    #     print(f"Normalizing activations for {k} with {v.cfg.normalize_activations}")
    #     print(f"Apply fine tuning scaling factor for {k} with {v.cfg.finetuning_scaling_factor}")

    for k in dictionaries:
        dictionaries[k].to(name2mod[k].module.cfg.device)

    return dictionaries

# TODO
class SanityChecks:
    def __init__(self, model, name2mod, dictionaries):
        self.model = model
        self.name2mod = name2mod
        self.dictionaries = dictionaries

    def SAE_setup(self):
        # Check for L1, L0, Variance Explained
        ...

    def solve_task(self, task):
        # Check that the model can solve the task. Otherwise, we are looking for something that is not there.
        ...
    
    def multi_GPU(self):
        # Check that forward passes and all operations are correctly handled in the current configuration.
        # TODO : print all GPU used and their usage + run some dummy passes to check that everything is working.
        ...
    
    def check_all(self, task):
        self.SAE_setup()
        self.solve_task(task)
        self.multi_GPU()