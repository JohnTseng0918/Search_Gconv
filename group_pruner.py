import torch
import torch.nn.utils.prune as prune

class GroupPruner(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        #print(mask.shape)
        return mask

def gconv_prune(module, name):
    GroupPruner.apply(module, name)
    return module
