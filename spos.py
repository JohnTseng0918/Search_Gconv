import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import utils
import random
from thop import profile
from pytorchcv.model_provider import get_model as ptcv_get_model

class SuperNet:
    def __init__(self, args):
        self.dataset = args.dataset
        self.train_batch_size = args.train_batch_size
        self.validate_batch_size = args.validate_batch_size
        self.epoch = args.epoch
        self.genome_type=[]
        self.genome_idx_type=[]

    def load_model(self):
        self.model = ptcv_get_model("resnet164bn_cifar100", pretrained=True)

    def build_oneshot(self):
        self.group_mod_list = nn.ModuleList()
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                g_list = utils.get_groups_choice_list(mod.in_channels, mod.out_channels)
                if len(g_list)>=3:
                    g_list=g_list[:2]
                sub_mod_list = nn.ModuleList()
                for g in g_list:
                    new_mod = nn.Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size, padding = mod.padding, bias = mod.bias, groups = g)
                    sub_mod_list.append(new_mod)
                self.group_mod_list.append(sub_mod_list)

    def pretrained_to_oneshot(self):
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

    def random_model_train_validate(self):
        self.random_model()
        self.print_genome()
        #self.lock_model()
        #self.partial_lock_model()
        self.train_one_epoch()
        #self.validate()

    def show_lock_status(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)

    def lock_model(self):
        for name, mod in self.model.named_modules():
            mod.requires_grad_(False)
    
    def unlock_model(self):
        for name, mod in self.model.named_modules():
            mod.requires_grad_(True)

    def partial_lock_model(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                if mod.groups==1:
                    mod.requires_grad_(False)

    def count_flops_params(self):
        input = torch.randn(1, 3, 32, 32).cuda()
        self.model.cuda()
        macs, params = profile(self.model, inputs=(input, ))
        print("MACs:", macs)
        print("params:", params)
        del input
        self.model.cpu()


    def print_genome(self):
        print(self.genome_type)

    def print_model(self):
        print(self.model)

    def train_one_epoch(self):
        self.model.cuda()
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(),weight_decay=0.0001)

        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        for i, (inputs, labels) in enumerate(self.trainloader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            n = inputs.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        print("top1 acc:", top1.avg)
        print("top5 acc:", top5.avg)
        print("avg loss:", losses.avg)
        print("-------------------------------------------------")
        self.model.cpu()
        self.update_random_model_weight()


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
            self.model.train()
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

    def train_supernet_lock(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())

        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        for inputs, labels in self.trainloader:
            self.random_model()
            self.partial_lock_model()
            self.model.train()
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
            self.unlock_model()

        print("train:")
        print("top1 acc:", top1.avg)
        print("top5 acc:", top5.avg)
        print("avg loss:", losses.avg)
        print("-------------------------------------------------")