import os
import torch
from torch.nn import Identity

class Submod:
    """
    A Submod is a helper class which contains a submodule of a model, along with an optional LayerNorm module
    that is applied before the submodule.
    """
    def __init__(self, name, module, LN_pre=None, LN_post=None):
        self.name = name
        self.module = module
        self.LN_pre = LN_pre if LN_pre is not None else Identity()
        self.LN_post = LN_post if LN_post is not None else Identity()

def save_circuit(save_dir, circuit, num_examples, dataset_name=None, model_name=None, threshold=None):
    save_dict = {
        "circuit" : dict(circuit),
    }
    threshold = str(threshold) if threshold is not None else 'None'
    threshold = threshold.replace('.', '_')

    if dataset_name is not None:
        save_basename = f"{dataset_name}_{model_name}_{threshold}_n{num_examples}"
    else:
        save_basename = f"{num_examples}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}{save_basename}.pt', 'wb') as outfile:
        torch.save(save_dict, outfile)
