import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import utils
import random
from models.resnet import *

class SuperNet:
    def __init__(self, args):
        self.dataset = args.dataset
        self.train_batch_size = args.train_batch_size
        self.validate_batch_size = args.validate_batch_size
        self.epoch = args.epoch
        self.genome_type=[]
        self.genome_idx_type=[]

    def load_model(self):
        self.model = cifar_resnet20(pretrained=True)

    def build_oneshot(self):
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
                    nn.init.xavier_normal_(new_mod.weight)
                    if g == 1:
                        new_mod.weight = torch.nn.Parameter(mod.weight)
                        new_mod.requires_grad_(False)
                    sub_mod_list.append(new_mod)
                self.group_mod_list.append(sub_mod_list)

    def warnup_oneshot(self):
        idx=0
        for name, mod in self.model.named_modules():
            mod.requires_grad_(False)

        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                if (len(self.group_mod_list[idx])==1):
                    idx+=1
                    continue
                for choice in self.group_mod_list[idx]:
                    if choice.groups==1:
                        continue
                    mod.weight = torch.nn.Parameter(choice.weight)
                    mod.groups = choice.groups
                    mod.requires_grad_(True)
                    print("idx:", idx)
                    print(choice)
                    self.validate()
                    self.train(optim.Adam(mod.parameters()))
                    self.validate()
                    mod.requires_grad_(False)
                    choice.weight = torch.nn.Parameter(mod.weight)

                mod.weight = torch.nn.Parameter(self.group_mod_list[idx][0].weight)
                mod.groups = 1
                mod.requires_grad_(False)

                idx+=1
        
        for name, mod in self.model.named_modules():
            mod.requires_grad_(True)

    def self_define_model(self):
        idx = 0
        self.genome_type=[]
        self.genome_idx_type=[]
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                t = 0
                if idx == 5:
                    t = 2
                mod.weight = torch.nn.Parameter(self.group_mod_list[idx][t].weight)
                mod.groups = self.group_mod_list[idx][t].groups
                self.genome_type.append(mod.groups)
                self.genome_idx_type.append(t)
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

    def print_genome(self):
        print(self.genome_type)

    def print_model(self):
        print(self.model)

    def train(self, optimizer, criterion = nn.CrossEntropyLoss()):
        utils.train(self.trainloader, self.model, optimizer, criterion, self.epoch)

    def validate(self):
        criterion = nn.CrossEntropyLoss()
        acc1, acc5, loss = utils.validate(self.testloader, self.model, criterion)
        print("validate:")
        print("top1 acc:", acc1)
        print("top5 acc:", acc5)    
        print("avg loss:", loss)
        print("-------------------------------------------------")

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

    def train_supernet(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())

        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        for inputs, labels in self.trainloader:
            self.random_model()
            self.model.cuda()

            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            self.update_random_model_weight()

            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            n = inputs.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        print("train:")
        print("top1 acc:", top1.avg)
        print("top5 acc:", top5.avg)
        print("avg loss:", losses.avg)
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