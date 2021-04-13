import torch
import torchvision
import torch.nn as nn
import random
import utils
from models.resnet import *

class OneShot:
    def __init__(self, args):
        self.model = cifar_resnet56()
        self.dataset = args.dataset
        self.train_batch_size = args.train_batch_size
        self.validate_batch_size = args.validate_batch_size

    def load_model(self):
        self.model.load_state_dict(torch.load("./warmup.pth"))
        
    def build_oneshot(self):
        self.group_mod_list = nn.ModuleList()
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                g_list = utils.get_groups_choice_list(mod.in_channels, mod.out_channels)
                sub_mod_list = nn.ModuleList()
                for g in g_list:
                    new_mod = nn.Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size, padding = mod.padding, bias = mod.bias, groups = g)
                    
                    # load weight
                    outc_start, intc_start = 0, 0
                    outc_interval, intc_interval = int(mod.out_channels/g), int(mod.in_channels/g)
                    w = torch.Tensor()
                    for i in range(g):
                        tmpweight = mod.weight[outc_start:outc_start+outc_interval, intc_start:intc_start+intc_interval,:,:]
                        w = torch.cat((w, tmpweight), 0)
                    new_mod.weight = torch.nn.Parameter(w)
                    
                    sub_mod_list.append(new_mod)
                self.group_mod_list.append(sub_mod_list)

    def random_model(self):
        idx = 0
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                t = random.randint(0, len(self.group_mod_list[idx])-1)
                mod.weight = self.group_mod_list[idx][t].weight
                mod.groups = self.group_mod_list[idx][t].groups
                #print(name, mod)
                idx+=1

    def print_model_conv2d(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                print(name, mod)

    def test_model(self):
        inputs = torch.randn((64,3,32,32))
        outputs = self.model(inputs)
        print(outputs.shape)

    def validate(self):
        criterion = nn.CrossEntropyLoss()
        acc1, acc5, loss = utils.validate(self.testloader, self.model, criterion)
        print("validate:")
        print("top1 acc:", acc1)
        print("top5 acc:", acc5)    
        print("avg loss:", loss)
        print("-------------------------------------------------")

    def get_dataloader(self):
        if self.dataset == "cifar10":
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            self.train_set = torchvision.datasets.CIFAR10(root='./data/cifar10',train=True,transform=train_transforms,download=True)
            self.test_set = torchvision.datasets.CIFAR10(root='./data/cifar10',train=False,transform=test_transforms,download=True)
        elif self.dataset == "cifar100":
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
            test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
            self.train_set = torchvision.datasets.CIFAR100(root='./data/cifar100',train=True,transform=train_transforms,download=True)
            self.test_set = torchvision.datasets.CIFAR100(root='./data/cifar100',train=False,transform=test_transforms,download=True)
        elif self.dataset == "imagenet":
            pass

        self.trainloader = torch.utils.data.DataLoader(self.train_set, batch_size=self.train_batch_size,shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_set, batch_size=self.validate_batch_size,shuffle=False)