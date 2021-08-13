import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import reduce

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

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)


class Bottleneck(nn.Module):
    """ Bottleneck module used in CondenseNet. """

    def __init__(self,
                 in_channels,
                 expansion=4,
                 growth_rate=12,
                 drop_rate=0):
        """ CTOR.
        Args:
          in_channels(int)
          expansion(int)
          growth_rate(int): the k value
          drop_rate(float): the dropout rate
        """
        super().__init__()

        # the input channels to the second 3x3 convolution layer
        channels = expansion * growth_rate

        # conv1: C -> 4 * k (according to the paper)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.choice0 = nn.ModuleList()
        self.choice0.append(conv1x1(in_channels, channels))
        #self.conv1 = conv1x1(in_channels, channels)
        # conv2: 4 * k -> k
        self.bn2 = nn.BatchNorm2d(channels)
        self.choice1 = nn.ModuleList()
        self.choice1.append(conv3x3(channels, growth_rate))
        # self.conv2 = conv3x3(channels, growth_rate)
        self.relu = nn.ReLU(inplace=True)

        self.drop_rate = drop_rate

    def forward(self, x, arch):
        """ forward """
        # conv1
        out = self.bn1(x)
        out = self.relu(out)
        # dropout (This is the different part)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.choice0[arch[0]](out)
        #out = self.conv1(out)
        # conv2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.choice1[arch[1]](out)
        #out = self.conv2(out)

        # concatenate results
        out = torch.cat((x, out), 1)
        return out

    def grow_choice(self):
        outchannel, inchannel, _, _ = self.choice0[0].weight.shape
        g_list = get_groups_choice_list(inchannel, outchannel)
        if len(g_list) > len(self.choice0):
            conv = conv1x1(inchannel, outchannel, self.choice0[0].stride, g_list[len(self.choice0)])
            self.choice0.append(conv)

        outchannel, inchannel, _, _ = self.choice1[0].weight.shape
        g_list = get_groups_choice_list(inchannel, outchannel)
        if len(g_list) > len(self.choice1):
            conv = conv3x3(inchannel, outchannel, self.choice1[0].stride, g_list[len(self.choice1)], 1)
            self.choice1.append(conv)

        return tuple([len(self.choice0), len(self.choice1)])

    def grow_choice_with_pretrained(self):
        outchannel, inchannel, _, _ = self.choice0[0].weight.shape
        g_list = get_groups_choice_list(inchannel, outchannel)
        if len(g_list) > len(self.choice0):
            g = g_list[len(self.choice0)]
            conv = conv1x1(inchannel, outchannel, self.choice0[0].stride, g)
            # move groups = 1 weight to groups = n
            W = self.choice0[0].weight
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
            self.choice0.append(conv)

        outchannel, inchannel, _, _ = self.choice1[0].weight.shape
        g_list = get_groups_choice_list(inchannel, outchannel)
        if len(g_list) > len(self.choice1):
            g = g_list[len(self.choice1)]
            conv = conv3x3(inchannel, outchannel, self.choice1[0].stride, g, 1)
            # move groups = 1 weight to groups = n
            W = self.choice1[0].weight
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
            self.choice1.append(conv)

        return tuple([len(self.choice0), len(self.choice1)])


class DenseBlock(nn.Sequential):
    """ Handles the logic of creating Dense layers """

    def __init__(self, num_layers, in_channels, growth_rate):
        """ CTOR.
        Args:
          num_layers(int): from stages
          growth_rate(int): from growth
        """
        super().__init__()

        for i in range(num_layers):
            layer = Bottleneck(in_channels + i * growth_rate, growth_rate=growth_rate)
            self.add_module('denselayer_%d' % (i + 1), layer)


class Transition(nn.Module):
    """ CondenseNet's transition, no convolution involved """

    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, arch):
        x = self.pool(x)
        return x

class condensenet86_oneshot(nn.Module):
    """ Main function to initialise CondenseNet. """

    def __init__(self, num_classes=10):
        """ CTOR.
        Args:
          stages(list): per layer depth
          growth(list): per layer growth rate
        """
        super().__init__()

        self.stages = [14, 14, 14]
        self.growth = [8, 16, 32]
        assert len(self.stages) == len(self.growth)

        # NOTE(): we removed the imagenet related branch
        self.init_stride = 1
        self.pool_size = 8

        
        # Initial nChannels should be 3
        # NOTE: this is a variable that traces the output size
        self.num_features = 2 * self.growth[0]

        # Dense-block 1 (224x224)
        # NOTE: this block will not be turned into a GConv
        self.init_conv = nn.Conv2d(3, self.num_features, kernel_size=3, stride=self.init_stride,padding=1, bias=False)
        self.features = nn.ModuleList()
        for i in range(len(self.stages)):
            # Dense-block i
            self.add_block(i)

        self.classifier = nn.Linear(self.num_features, num_classes)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def add_block(self, i):
        # Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i])

        for idx in range(len(block)):
            self.features.append(block[idx])

        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = Transition(in_channels=self.num_features)
            self.features.append(trans)
        else:
            self.features.append(nn.BatchNorm2d(self.num_features))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(nn.AvgPool2d(self.pool_size))

    def forward(self, x, architecture):
        x = self.init_conv(x)

        for i, arch_id in enumerate(architecture[:-3]):
            x = self.features[i](x, arch_id)
        x = self.features[-3](x)
        x = self.features[-2](x)
        x = self.features[-1](x)

        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out

    def grow(self):
        self.all_choice_list = []
        for i in range(len(self.features)):
            if isinstance(self.features[i] , Bottleneck):
                n = self.features[i].grow_choice()
                self.all_choice_list.append(n)
            else:
                self.all_choice_list.append(1)

    def grow_with_pretrained(self):
        self.all_choice_list = []
        for i in range(len(self.features)):
            if isinstance(self.features[i] , Bottleneck):
                n = self.features[i].grow_choice_with_pretrained()
                self.all_choice_list.append(n)
            else:
                self.all_choice_list.append(1)
    

    def get_all_arch(self):
        return tuple(self.all_choice_list)

    def get_origin_arch(self):
        num_feature = len(self.features)
        res = tuple([0] * num_feature)
        return res