from pytorchcv.models.fastscnn import FastPyramidPooling
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import utils
import time
import random
from data_loader import get_train_valid_loader, get_test_loader
from pytorchcv.model_provider import get_model as ptcv_get_model
from thop import profile

class SuperNetEA:
    def __init__(self, args):
        self.dataset = args.dataset
        if self.dataset == "cifar100":
            self.num_class=100
        elif self.dataset == "cifar10":
            self.num_class=10
        elif self.dataset == "imagenet":
            self.num_class=1000
        self.arch = args.arch
        self.train_batch_size = args.train_batch_size
        self.validate_batch_size = args.validate_batch_size
        self.genome_type=[]
        self.genome_idx_type=[]
        self.FLOPs = args.FLOPs
        self.params = args.params
        self.population = args.population
        self.search_epoch = args.search_epoch
        self.topk_num = args.topk_num
        #self.criterion = utils.CrossEntropyLabelSmooth(self.num_class, 0.1)
        self.criterion = nn.CrossEntropyLoss()

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
                nn.init.xavier_uniform_(new_mod.weight)
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
    
    def check_grow_fully(self):
        for i in self.not_been_built_list:
            if len(i)!=0:
                return False
        return True

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
                    nn.init.xavier_uniform_(new_mod.weight)
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
    
    def pretrained_to_all_supernet(self):
        idx = 0
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                for choice in self.group_mod_list[idx]:
                    if choice.groups == mod.groups:
                        choice.weight = torch.nn.Parameter(mod.weight)
                    else:
                        g = choice.groups
                        W = mod.weight
                        outc_start, intc_start = 0, 0
                        outc_interval = int(n/g)
                        intc_interval = int(c/g)
                        tensorlist=[]
                        for i in range(g):
                            tensorlist.append(W[outc_start:outc_start+outc_interval, intc_start:intc_start+intc_interval,:,:])
                            outc_start+=outc_interval
                            intc_start+=intc_interval
                        wnew = torch.cat(tuple(tensorlist),0)
                        choice.weight = torch.nn.Parameter(wnew)
                idx+=1

    def permute_model(self, n_sort=10):
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                n, c, h, w = mod.weight.shape
                if h==1 or w ==1:
                    continue
                g_list = utils.get_groups_choice_list(mod.in_channels, mod.out_channels)
                if len(g_list) > 1:
                    self.model.cpu()
                    w = mod.weight.detach()
                    w = utils.get_permute_weight(w, g_list[1], n_sort)
                    mod.weight = torch.nn.Parameter(w)
                    self.train_one_epoch()
                    self.model.cpu()

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

    def search(self):
        # init EA get random population (with constrain)
        populist = []
        self.topk = []
        count=0
        while count < self.population:
            self.random_model()
            isvalid = self.check_constrain()
            if isvalid == True:
                count+=1
                populist.append(tuple(self.genome_idx_type))
        
        n = int(len(populist) / 2)
        m = int(len(populist) / 2)
        prob = 0.1
        for i in range(self.search_epoch):
            #inference
            for p in populist:
                self.genome_build_model(p)
                self.print_genome()
                acc1, _, _ = self.validate()
                self.topk.append((p,acc1))
            
            #update topk
            self.topk = list(set(self.topk))
            self.topk.sort(key=lambda s:s[1], reverse=True)
            if len(self.topk) >= self.topk_num:
                self.topk = self.topk[:self.topk_num]
            print("topk:",self.topk)

            #crossover
            crossover_child = []
            max_iter = n * 10
            while max_iter > 0 and len(crossover_child) < n:
                max_iter-=1
                s1, s2 = random.sample(self.topk, 2)
                p1, _ = s1
                p2, _ = s2
                child = [random.choice([i,j]) for i,j in zip(p1,p2)]
                self.genome_build_model(child)
                if self.check_constrain():
                    crossover_child.append(tuple(child))
            

            #mutation
            mutation_child = []
            max_iter = m * 10
            while max_iter > 0 and len(mutation_child) < m:
                max_iter-=1
                s1 = random.choice(self.topk)
                p1, _ = s1
                p2 = []
                for i in range(len(p1)):
                    if random.random() < prob:
                        p2.append(random.randint(0,len(self.group_mod_list[i])-1))
                    else:
                        p2.append(p1[i])
                self.genome_build_model(p2)
                if self.check_constrain():
                    mutation_child.append(tuple(p2))
            

            #union
            populist = []
            populist = crossover_child + mutation_child
            n = int(len(populist) / 2)
            m = int(len(populist) / 2)
    
    def select_from_topk(self, k=0):
        m, _ = self.topk[k]
        return m

    def check_constrain(self):
        macs, params = self.count_flops_params()
        if self.params != None and self.params <= params:
            return False
        if self.FLOPs != None and self.FLOPs <= macs:
            return False
        return True

    def count_flops_params(self):
        self.model.cpu()
        if self.dataset == "cifar10" or self.dataset =="cifar100":
            inputs = torch.randn(1, 3, 32, 32)
        else:
            inputs = torch.randn(1, 3, 224, 224)
        macs, params = profile(self.model, inputs = (inputs,), verbose=False)
        return macs, params
    
    def measure_latency(self):
        self.model.cpu()
        inputs = torch.randn(16, 3, 32, 32)
        self.model.eval()
        latlist=[]
        for i in range(50):
            torch.cuda.synchronize()
            start = time.time()
            self.model(inputs)
            torch.cuda.synchronize()
            end = time.time()
            latlist.append(end-start)
        avg_time = sum(latlist[20:])/len(latlist[20:])
        print(avg_time)
        self.model.cuda()
        return avg_time

    def print_genome(self):
        print(self.genome_type)

    def print_idx_genome(self):
        print(self.genome_idx_type)

    def unlock_model(self):
        for name, mod in self.model.named_modules():
            mod.requires_grad_(True)

    def train_one_epoch(self):
        optimizer = optim.Adam(self.model.parameters())
        acc1, acc5, loss = utils.train_one_epoch(self.trainloader, self.model, optimizer, self.criterion)
        print("train:")
        print("top1 acc:", acc1)
        print("top5 acc:", acc5)
        print("avg loss:", loss)
        print("-------------------------------------------------")

    def fine_tune(self, lr=0.005, ftepoch=100, momentum=0.9, testepoch=100):
        isnesterov=True
        if momentum==0:
            isnesterov=False
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=0.001, nesterov=isnesterov)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ftepoch)
        for e in range(ftepoch):
            acc1, acc5, loss = utils.train_one_epoch(self.trainloader, self.model, optimizer, self.criterion)
            print("epoch", e+1, ":")
            print("top1 acc:", acc1)
            print("top5 acc:", acc5)
            print("avg loss:", loss)
            print("-------------------------------------------------")
            scheduler.step()
            if e+1 >= testepoch:
                self.test()

    def validate(self):
        acc1, acc5, loss = utils.validate(self.validateloader, self.model, self.criterion)
        print("validate:")
        print("top1 acc:", acc1)
        print("top5 acc:", acc5)    
        print("avg loss:", loss)
        print("-------------------------------------------------")
        return acc1, acc5, loss

    def test(self):
        acc1, acc5, loss = utils.validate(self.testloader, self.model, self.criterion)
        print("test:")
        print("top1 acc:", acc1)
        print("top5 acc:", acc5)    
        print("avg loss:", loss)
        print("-------------------------------------------------")
        return acc1, acc5, loss

    def get_dataloader(self):
        self.trainloader, self.validateloader = get_train_valid_loader("./data/cifar100", self.train_batch_size, augment=True, random_seed=87)
        self.testloader = get_test_loader("./data/cifar100", self.validate_batch_size, shuffle=False)

    def random_model_valid(self, timeout=10):
        for i in range(timeout):
            self.random_model()
            _, params = self.count_flops_params()
            if params >= self.params*0.9 and params <= self.params*1.1:
                return
        self.random_model()
        
    def train_supernet(self, epoch, lr=0.1):
        dynamic_lr = lr
        for e in range(epoch):
            if e % 10==0 and e!=0:
                dynamic_lr=dynamic_lr*0.5
            print("epoch:", e+1, "train supernet:")
            if e <= epoch*0.9:
                self.train_supernet_one_epoch(valid=False, lr=dynamic_lr)
            else:
                self.train_supernet_one_epoch(valid=True, lr=dynamic_lr)

    def train_supernet_one_epoch(self, valid=False, lr=0.001):
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        for inputs, labels in self.trainloader:
            if valid==True:
                self.random_model_valid()
            else:
                self.random_model()
            self.model.train()
            self.model.cuda()
            optimizer = optim.SGD(self.model.parameters(),lr=lr,weight_decay=0.0001)

            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            self.update_random_model_weight()

            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            n = inputs.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            
        print("top1 acc:", top1.avg)
        print("top5 acc:", top5.avg)
        print("avg loss:", losses.avg)
        print("-------------------------------------------------")