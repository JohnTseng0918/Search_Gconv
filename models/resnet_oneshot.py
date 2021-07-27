import torch
from torch import Tensor
import torch.nn as nn
import utils
from typing import Type, Any, Callable, Union, List, Optional

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.choice = nn.ModuleList()
        self.choice.append(conv3x3(width, width, stride, groups, dilation))
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def grow_choice(self):
        outchannel, inchannel, _, _ = self.choice[0].weight.shape
        g_list = utils.get_groups_choice_list(inchannel, outchannel)
        if len(g_list) > len(self.choice):
            conv = conv3x3(inchannel, outchannel, self.stride, g_list[len(self.choice)], 1)
            self.choice.append(conv)
        return len(self.choice)

    def forward(self, x: Tensor, arch) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.choice[arch](out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class resnet50_oneshot(nn.Module):
    def __init__(self):
        super(resnet50_oneshot, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        num_classes = 1000

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = torch.nn.ModuleList()        
        layer1 = self._make_layer(Bottleneck, 64, 3)
        layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        for i in range(len(layer1)):
            self.features.append(layer1[i])
        for i in range(len(layer2)):
            self.features.append(layer2[i])
        for i in range(len(layer3)):
            self.features.append(layer3[i])
        for i in range(len(layer4)):
            self.features.append(layer4[i])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, architecture):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, arch_id in enumerate(architecture):
            x = self.features[i](x, arch_id)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def grow(self):
        self.all_choice_list = []
        for i in range(len(self.features)):
            n = self.features[i].grow_choice()
            self.all_choice_list.append(n)
    
    def get_all_arch(self):
        return tuple(self.all_choice_list)



class resnet164_oneshot(nn.Module):
    def __init__(self, num_classes = 100):
        super(resnet164_oneshot, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.inplanes = 16
        self.groups = 1
        self.base_width = 16
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.features = torch.nn.ModuleList()        
        layer1 = self._make_layer(Bottleneck, 16, 18)
        layer2 = self._make_layer(Bottleneck, 32, 18, stride=2)
        layer3 = self._make_layer(Bottleneck, 64, 18, stride=2)
        for i in range(len(layer1)):
            self.features.append(layer1[i])
        for i in range(len(layer2)):
            self.features.append(layer2[i])
        for i in range(len(layer3)):
            self.features.append(layer3[i])
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, architecture):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i, arch_id in enumerate(architecture):
            x = self.features[i](x, arch_id)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def grow(self):
        self.all_choice_list = []
        for i in range(len(self.features)):
            n = self.features[i].grow_choice()
            self.all_choice_list.append(n)
    
    def get_all_arch(self):
        return tuple(self.all_choice_list)