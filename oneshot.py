import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import random
import utils
from models.resnet import *

class OneShot:
    def __init__(self, args):
        self.model = cifar_resnet56()
        self.dataset = args.dataset
        self.train_batch_size = args.train_batch_size
        self.validate_batch_size = args.validate_batch_size
        self.genome_type=[]
        self.genome_idx_type=[]

    def load_model(self, PATH):
        self.model.load_state_dict(torch.load(PATH))
        
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
        self.genome_type=[]
        self.genome_idx_type=[]
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                t = random.randint(0, len(self.group_mod_list[idx])-1)
                mod.weight = torch.nn.Parameter(self.group_mod_list[idx][t].weight)
                mod.groups = self.group_mod_list[idx][t].groups
                self.genome_type.append(mod.groups)
                self.genome_idx_type.append(t)
                idx+=1

    def update_random_model_weight(self):
        idx = 0
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                t = self.genome_idx_type[idx]
                self.group_mod_list[idx][t].weight = torch.nn.Parameter(mod.weight)
                idx+=1

    def print_model_conv2d(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                print(name, mod)

    def test_model(self):
        inputs = torch.randn((64,3,32,32))
        outputs = self.model(inputs)
        print(outputs.shape)
    
    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())

        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        for inputs, labels in self.trainloader:
            self.random_model()
            #self.show_genome_type()
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


    def validate(self):
        criterion = nn.CrossEntropyLoss()
        acc1, acc5, loss = utils.validate(self.testloader, self.model, criterion)
        print("validate:")
        print("top1 acc:", acc1)
        print("top5 acc:", acc5)    
        print("avg loss:", loss)
        print("-------------------------------------------------")

    def show_genome_type(self):
        print(self.genome_type)

    def show_genome_idx_type(self):
        print(self.genome_idx_type)

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