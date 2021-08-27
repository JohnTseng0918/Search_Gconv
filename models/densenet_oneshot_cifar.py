from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import reduce
import math

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

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, groups=1):
        super(Conv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.choice = nn.ModuleList()
        self.choice.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding, bias=False,
                                    groups=groups))

    def forward(self, x, arch):
        x = self.norm(x)
        x = self.relu(x)
        x = self.choice[arch](x)
        return x

    def grow_chioce(self):
        outchannel, inchannel, _, _ = self.choice[0].weight.shape
        g_list = get_groups_choice_list(inchannel, outchannel)
        if len(g_list) > len(self.choice):
            conv = nn.Conv2d(inchannel, outchannel,
                                    kernel_size=self.choice[0].kernel_size,
                                    stride=self.choice[0].stride,
                                    padding=self.choice[0].padding, bias=False,
                                    groups=g_list[len(self.choice)])
            self.choice.append(conv)
        return len(self.choice)

    def grow_choice_with_pretrained(self):
        outchannel, inchannel, _, _ = self.choice[0].weight.shape
        g_list = get_groups_choice_list(inchannel, outchannel)
        if len(g_list) > len(self.choice):
            g = g_list[len(self.choice)]
            conv = nn.Conv2d(inchannel, outchannel,
                                    kernel_size=self.choice[0].kernel_size,
                                    stride=self.choice[0].stride,
                                    padding=self.choice[0].padding, bias=False,
                                    groups=g_list[len(self.choice)])
            W = self.choice[0].weight
            outc_start, intc_start = 0, 0
            outc_interval = int(outchannel/g)
            intc_interval = int(inchannel/g)
            tensorlist=[]
            for i in range(g):
                tensorlist.append(W[outc_start:outc_start+outc_interval, intc_start:intc_start+intc_interval,:,:])
                outc_start+=outc_interval
                intc_start+=intc_interval
            wnew = torch.cat(tuple(tensorlist),0)
            conv.weight = torch.nn.Parameter(wnew)
            self.choice.append(conv)
        return len(self.choice)

def make_divisible(x, y):
    return int((x // y + 1) * y) if x % y else int(x)

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, group_1x1, group_3x3, bottleneck):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = group_1x1
        self.group_3x3 = group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = Conv(in_channels, bottleneck * growth_rate,
                           kernel_size=1, groups=self.group_1x1)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x, arch):
        x_ = x
        x = self.conv_1(x, arch[0])
        x = self.conv_2(x, arch[1])
        return torch.cat([x_, x], 1)
    
    def grow_choice(self):
        len1 = self.conv_1.grow_chioce()
        len2 = self.conv_2.grow_chioce()
        return(len1, len2)
    
    def grow_choice_with_pretrained(self):
        len1 = self.conv_1.grow_choice_with_pretrained()
        len2 = self.conv_2.grow_choice_with_pretrained()
        return(len1, len2)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, group_1x1, group_3x3, bottleneck):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, group_1x1, group_3x3, bottleneck)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels, group_1x1):
        super(_Transition, self).__init__()
        self.conv = Conv(in_channels, out_channels,
                         kernel_size=1, groups=group_1x1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, arch):
        x = self.conv(x, arch)
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, stages, growth, num_classes, data, bottleneck, group_1x1, group_3x3, reduction):

        super(DenseNet, self).__init__()

        self.stages = stages
        self.growth = growth
        self.reduction = reduction
        self.bottleneck = bottleneck
        self.group_1x1 = group_1x1
        self.group_3x3 = group_3x3
        assert len(self.stages) == len(self.growth)
        #self.args = args
        if data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        ### Set initial width to 2 x growth_rate[0]
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.init_conv = nn.Conv2d(3, self.num_features,kernel_size=3,stride=self.init_stride,padding=1,bias=False)
        self.features = nn.ModuleList()

        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)
        ### Linear layer
        self.classifier = nn.Linear(self.num_features, num_classes)

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def add_block(self, i):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            group_1x1=self.group_1x1, 
            group_3x3=self.group_3x3, 
            bottleneck=self.bottleneck
        )
        #self.features.add_module('denseblock_%d' % (i + 1), block)
        for idx in range(len(block)):
            self.features.append(block[idx])
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            out_features = make_divisible(math.ceil(self.num_features * self.reduction),
                                          self.group_1x1)
            trans = _Transition(in_channels=self.num_features,
                                out_channels=out_features,
                                group_1x1=self.group_1x1)
            self.features.append(trans)
            self.num_features = out_features
        else:
            self.features.append(nn.BatchNorm2d(self.num_features))
            self.features.append( nn.ReLU(inplace=True))
            ### Use adaptive ave pool as global pool
            self.features.append(nn.AvgPool2d(self.pool_size))

    def forward(self, x, architecture):
        features = self.init_conv(x)

        for i, arch_id in enumerate(architecture[:-3]):
            features = self.features[i](features, arch_id)
        features = self.features[-3](features)
        features = self.features[-2](features)
        features = self.features[-1](features)

        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
    
    def grow(self):
        self.all_choice_list = []
        for i in range(len(self.features)):
            if isinstance(self.features[i] , _DenseLayer):
                n = self.features[i].grow_choice()
                self.all_choice_list.append(n)
            else:
                self.all_choice_list.append(1)
    
    def grow_with_pretrained(self):
        self.all_choice_list = []
        for i in range(len(self.features)):
            if isinstance(self.features[i] , _DenseLayer):
                n = self.features[i].grow_choice_with_pretrained()
                self.all_choice_list.append(n)
            else:
                self.all_choice_list.append(1)
    
    def get_all_arch(self):
        return tuple(self.all_choice_list)
    
    def get_origin_arch(self):
        res=[]
        for i in self.features:
            if isinstance(i, _DenseLayer):
                res.append((0, 0))
            else:
                res.append(0)
        return tuple(res)

def DenseNet_BC_100_k_12(num_classes=100,data="cifar100"):
    return DenseNet(stages=[16,16,16],growth=[12,12,12],num_classes=num_classes,data=data,bottleneck=4,group_1x1=1,group_3x3=1,reduction=0.5)