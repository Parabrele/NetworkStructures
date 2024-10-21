import torch
from utils.activation import SparseAct

zero_ablation = lambda x: x.zeros_like()

mean_ablation = lambda x: x.mean(dim=(0, 1)).expand_as(x)

id_ablation = lambda x: x

def resample_ablation(x):
    """
    Sample another input from the dataset and use it's activations for ablation
    """
    raise NotImplementedError

def random_ablation(x, dist='normal'):
    """
    Sample a random vector from a given distribution (default is normal) and use it for ablation
    """
    raise NotImplementedError