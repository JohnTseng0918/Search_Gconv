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
                sub_mod_list = nn.ModuleList()
                for g in g_list:
                    new_mod = nn.Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size, padding = mod.padding, bias = mod.bias, groups = g)
                    sub_mod_list.append(new_mod)
                self.group_mod_list.append(sub_mod_list)

    def init_supernet(self):
        self.get_all_group_choice_supernet()
        self.group_mod_list = nn.ModuleList()
        idx=0
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                sub_mod_list = nn.ModuleList()
                new_mod = nn.Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size, padding = mod.padding, bias = mod.bias, groups = mod.groups)
                sub_mod_list.append(new_mod)
                self.group_mod_list.append(sub_mod_list)
                self.not_been_built_list[idx].remove(new_mod.groups)
                idx+=1

    def get_all_group_choice_supernet(self):
        self.not_been_built_list = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                g_list = utils.get_groups_choice_list(mod.in_channels, mod.out_channels)
                self.not_been_built_list.append(g_list)

    def grow_supernet(self):
        idx=0
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                if len(self.not_been_built_list[idx])!=0:
                    g = self.not_been_built_list[idx][0]
                    new_mod = nn.Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size, padding = mod.padding, bias = mod.bias, groups = g)
                    self.group_mod_list[idx].append(new_mod)
                    self.not_been_built_list[idx].remove(g)
                idx+=1

    def update_random_model_weight(self):
        idx = 0
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                t = self.genome_idx_type[idx]
                self.group_mod_list[idx][t].weight = torch.nn.Parameter(mod.weight)
                idx+=1

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

    def random_model_train(self):
        self.random_model()
        self.print_genome()
        self.train_one_epoch()
        self.update_random_model_weight()

    def random_model_train_lock(self):
        self.random_model()
        self.print_genome()
        self.partial_lock_model()
        self.train_one_epoch()
        self.unlock_model()
        self.update_random_model_weight()


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

    def partial_lock_model(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                if mod.groups==1:
                    mod.requires_grad_(False)

    def unlock_model(self):
        for name, mod in self.model.named_modules():
            mod.requires_grad_(True)


    def train_one_epoch(self):
        optimizer = optim.Adam(self.model.parameters())
        acc1, acc5, loss = utils.train_one_epoch(self.trainloader, self.model, optimizer, nn.CrossEntropyLoss())
        print("train:")
        print("top1 acc:", acc1)
        print("top5 acc:", acc5)
        print("avg loss:", loss)
        print("-------------------------------------------------")

    def fine_tune(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.001, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
        utils.train(self.trainloader, self.model, optimizer, scheduler, nn.CrossEntropyLoss(), 150)

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


    def train_n_iteration(self, n=50):
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        self.model.cuda()
        self.model.train()

        for i, (inputs, labels) in enumerate(self.trainloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i==n-1:
                break