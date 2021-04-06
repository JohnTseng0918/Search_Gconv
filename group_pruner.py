import torch
import torch.nn.utils.prune as prune
from utils import get_random_groups

class GroupPrunerInit(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.fill_(1)
        return mask

def gconv_prune_init(module, name):
    GroupPrunerInit.apply(module, name)
    return module
