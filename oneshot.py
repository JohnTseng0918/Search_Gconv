import torch
import torch.nn as nn
from utils import get_groups_choice_list
from models.resnet import *

class OneShot:
    def __init__(self, args):
        self.model = cifar_resnet56()

    def load_model(self):
        self.model.load_state_dict(torch.load("./warmup.pth"))
        
    def build_oneshot(self):
        self.group_mod_list = nn.ModuleList()
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                g_list = get_groups_choice_list(mod.in_channels, mod.out_channels)
                sub_mod_list = nn.ModuleList()
                for g in g_list:
                    new_mod = nn.Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size, groups = g)
                    # TO DO:: load conv2d.weight
                    sub_mod_list.append(new_mod)
                self.group_mod_list.append(sub_mod_list)