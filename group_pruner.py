import torch
import torch.nn.utils.prune as prune
from utils import get_random_groups

class GroupPruner(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        n,c,h,w = t.shape
        g = get_random_groups(c, n)
        outc_start, intc_start = 0, 0
        outc_interval = int(n/g)
        intc_interval = int(c/g)
        mask.fill_(0)
        for i in range(g):
            mask[outc_start:outc_start+outc_interval, intc_start:intc_start+intc_interval,:,:] = 1
            outc_start+=outc_interval
            intc_start+=intc_interval
        return mask

def gconv_prune(module, name):
    GroupPruner.apply(module, name)
    return module
