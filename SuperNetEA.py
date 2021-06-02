import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import utils
import random
from pytorchcv.model_provider import get_model as ptcv_get_model
from thop import profile

class SuperNetEA:
    def __init__(self, args):
        self.dataset = args.dataset
        self.arch = args.arch
        self.train_batch_size = args.train_batch_size
        self.validate_batch_size = args.validate_batch_size
        self.epoch = args.epoch
        self.genome_type=[]
        self.genome_idx_type=[]

    def load_model(self):
        self.model = ptcv_get_model(self.arch, pretrained=True)

    def build_supernet(self):
        self.group_mod_list = nn.ModuleList()
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                g_list = utils.get_groups_choice_list(mod.in_channels, mod.out_channels)
                ## delete----
                if len(g_list)>=3:
                    g_list=g_list[:2]
                ## delete----
                sub_mod_list = nn.ModuleList()
                for g in g_list:
                    new_mod = nn.Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size, padding = mod.padding, bias = mod.bias, groups = g)
                    sub_mod_list.append(new_mod)
                self.group_mod_list.append(sub_mod_list)

    def pretrained_to_supernet(self):
        idx = 0
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                for choice in self.group_mod_list[idx]:
                    if choice.groups == mod.groups:
                        choice.weight = torch.nn.Parameter(mod.weight)
                idx+=1

    def genome_build_model(self, genome_list):
        idx = 0
        self.genome_type=[]
        self.genome_idx_type=[]
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue

                t = genome_list[idx]
                mod.weight = torch.nn.Parameter(self.group_mod_list[idx][t].weight)
                mod.groups = self.group_mod_list[idx][t].groups
                self.genome_type.append(mod.groups)
                self.genome_idx_type.append(t)
                idx+=1

    def random_model_train_validate(self):
        self.random_model()
        self.print_genome()
        self.train_one_epoch()
        self.validate()


    def random_model(self):
        idx = 0
        self.genome_type=[]
        self.genome_idx_type=[]
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                t = random.randint(0, len(self.group_mod_list[idx])-1)
                mod.weight = torch.nn.Parameter(self.group_mod_list[idx][t].weight)
                mod.groups = self.group_mod_list[idx][t].groups
                self.genome_type.append(mod.groups)
                self.genome_idx_type.append(t)
                idx+=1

    def count_flops_params(self):
        if self.dataset == "cifar10" or self.dataset =="cifar100":
            inputs = torch.randn(1, 3, 32, 32)
        else:
            inputs = torch.randn(1, 3, 224, 224)
        input = inputs.cuda()
        self.model.cuda()
        macs, params = profile(self.model, inputs=(input, ))
        print("MACs:", macs)
        print("params:", params)
        del input
        self.model.cpu()
        return macs, params

    def print_genome(self):
        print(self.genome_type)

    def train_one_epoch(self):
        optimizer = optim.Adam(self.model.parameters())
        utils.train_one_epoch(self.trainloader, self.model, optimizer, nn.CrossEntropyLoss())

    def fine_tune(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        utils.train(self.trainloader, self.model, optimizer, scheduler, nn.CrossEntropyLoss(), 100)

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