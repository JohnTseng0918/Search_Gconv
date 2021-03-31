import torch
import torch.nn as nn
from utils import get_random_groups

def conv3x3(in_planes, out_planes, stride=1, random_group=False):
    """3x3 convolution with padding"""
    if random_group==False:
      return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    else:
      g = get_random_groups(in_planes, out_planes)
      return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups = g, bias=False)


def conv1x1(in_planes, out_planes, stride=1, random_group=False):
    """1x1 convolution"""
    if random_group==False:
      return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    else:
      g = get_random_groups(in_planes, out_planes)
      return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups = g, bias=False)

class BasicBlock(nn.Module):
  """ The stacked convolution block design.

  Structure:
    [in_planes] -> conv3x3 -> [planes] -> conv3x3 -> [planes]
  """
  expansion = 1

  def __init__(self, in_planes, planes, stride=1, downsample=None, random_group=False, **kwargs):
    """ CTOR.
    
    Args:
      in_planes(int): input channels
      planes(int): final output channels
      stride(int)
      downsample(func): downsample function
      groups(int): number of groups
      indices(list): a permutation
      mask(bool): whether to build masked convolution
    """
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(in_planes, planes, stride=stride, random_group=random_group, **kwargs)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes, random_group=random_group, **kwargs)
    self.bn2 = nn.BatchNorm2d(planes)

    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual  # shortcut
    out = self.relu(out)

    return out

class Bottleneck(nn.Module):
  """ Bottleneck block with expansion > 1.
  
  Structure:
    [in_planes] -> conv1x1 -> [planes] -> conv3x3 -> [planes] -> conv1x1 -> [planes x 4]
  """
  expansion = 4

  def __init__(self, in_planes, planes, stride=1, downsample=None, random_group=False, **kwargs):
    """ CTOR.
    
    Args:
      in_planes(int): input channels
      planes(int): final output channels
      stride(int)
      downsample(func): downsample function
    """

    super(Bottleneck, self).__init__()
    self.conv1 = conv1x1(in_planes, planes, random_group=random_group, **kwargs)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes, stride=stride, random_group=random_group, **kwargs)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = conv1x1(planes, planes * self.expansion, random_group=random_group, **kwargs)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)

    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, random_group=False):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16, random_group=random_group)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0], random_group=random_group)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, random_group=random_group)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, random_group=random_group)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, random_group=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, random_group=random_group),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, random_group=random_group))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, random_group=random_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def cifar_resnet56(pretrained=None, random_group=False, **kwargs):
    if random_group==False:
      if pretrained is None:
          model = CifarResNet(BasicBlock, [9, 9, 9], num_classes=100)
      else:
          model = CifarResNet(BasicBlock, [9, 9, 9], num_classes=100)
          model.load_state_dict(torch.load("./pretrained/cifar100-resnet56.pth"))
    else:
      model = CifarResNet(BasicBlock, [9, 9, 9], num_classes=100, random_group=True)
    return model