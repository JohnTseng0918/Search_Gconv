import torch
import torch.nn as nn
from functools import reduce
import random
import numpy as np

def factors(n):
    """
    Copied from https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
    """    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def get_groups_choice_list(inchannel, outchannel):
    fin, fout = factors(inchannel), factors(outchannel)
    g_list = list(sorted(fin.intersection(fout)))
    return g_list

def get_random_groups(inchannel, outchannel):
    g_list = get_groups_choice_list(inchannel, outchannel)
    return random.choice(g_list)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(validate_loader, model, criterion):
    model.cuda()
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    
    for i, (inputs, labels) in enumerate(validate_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
        n = inputs.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    return top1.avg, top5.avg, losses.avg

def train_one_epoch(train_loader, model, optimizer, criterion):
    model.cuda()
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
        n = inputs.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    return top1.avg, top5.avg, losses.avg

def train(train_loader, model, optimizer, scheduler, criterion, epoch):
    for e in range(epoch):
        acc1, acc5, loss = train_one_epoch(train_loader, model, optimizer, criterion)
        print("epoch", e+1, ":")
        print("top1 acc:", acc1)
        print("top5 acc:", acc5)
        print("avg loss:", loss)
        print("-------------------------------------------------")
        scheduler.step()

def get_flops(model, input):
    list_conv = []

    def conv_hook(self, input, output):
        bias_ops = 1 if self.bias is not None else 0
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        assert self.in_channels % self.groups == 0

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels // self.groups)
        flops = 2 * output_height * output_width * (kernel_ops + bias_ops) * output_channels * batch_size
        
        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        total_mul = self.in_features
        num_elements = output.numel()
        flops = 2 * total_mul * num_elements

        list_linear.append(flops)
    
    list_bn = []
    def bn_hook(self, input, output):        
        params = self.num_features*2
        list_bn.append(params)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    out = model(input)

    total_flops = sum(sum(i) for i in [list_conv, list_linear, list_bn])
    return total_flops

def get_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_params_flops(model, input, arch):
    list_conv_param = []
    list_conv_flops = []

    def conv_hook(self, input, output):
        bias_ops = 1 if self.bias is not None else 0
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        assert self.in_channels % self.groups == 0

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels // self.groups)
        flops = 2 * output_height * output_width * (kernel_ops + bias_ops) * output_channels * batch_size
        params = self.weight.numel()
        list_conv_param.append(params)
        list_conv_flops.append(flops)
        
    list_linear_param = []
    list_linear_flops = []

    def linear_hook(self, input, output):
        total_mul = self.in_features
        num_elements = output.numel()
        flops = 2 * total_mul * num_elements
        params = self.weight.numel() + self.bias.numel()
        
        list_linear_param.append(params)
        list_linear_flops.append(flops)
    
    list_bn_param = []
    
    def bn_hook(self, input, output):        
        params = self.num_features*2
        list_bn_param.append(params)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    out = model(input, arch)

    total_params = sum(sum(i) for i in [list_conv_param, list_linear_param, list_bn_param])
    total_flops = sum(sum(i) for i in [list_conv_flops, list_linear_flops])
    
    return total_params, total_flops


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * \
            targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L363-L384
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)