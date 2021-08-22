from data_loader_noddp import get_train_valid_loader, get_test_loader
from pytorchcv.model_provider import get_model as ptcv_get_model
import utils
import torch
import torch.optim as optim
import torch.nn as nn
from models.resnet_oneshot_test import ResNet, Bottleneck

dataset = "imagenet"
path = "./data/" + dataset

model = ptcv_get_model("resnet50", pretrained=True)
oneshot_model = ResNet(Bottleneck,[3, 4, 6, 3])



testloader = get_test_loader(path, 32, shuffle=False, data=dataset)

conv_w_list = []
bn_w_list = []
bn_bias_list = []
linear_w_list = []
linear_bias_list = []
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.modules.conv.Conv2d):
        conv_w_list.append(mod.weight)
    if isinstance(mod, torch.nn.BatchNorm2d):
        bn_w_list.append(mod.weight)
        bn_bias_list.append(mod.bias)
    if isinstance(mod, torch.nn.Linear):
        linear_w_list.append(mod.weight)
        linear_bias_list.append(mod.bias)

convidx = 0
bn_idx = 0
linear_idx = 0
for name, mod in oneshot_model.named_modules():
    if isinstance(mod, torch.nn.modules.conv.Conv2d):
        mod.weight = torch.nn.Parameter(conv_w_list[convidx])
        convidx+=1
    if isinstance(mod, torch.nn.BatchNorm2d):
        mod.weight = torch.nn.Parameter(bn_w_list[bn_idx])
        mod.bias = torch.nn.Parameter(bn_bias_list[bn_idx])
        bn_idx+=1
    if isinstance(mod, torch.nn.Linear):
        mod.weight = torch.nn.Parameter(linear_w_list[linear_idx])
        mod.bias = torch.nn.Parameter(linear_bias_list[linear_idx])

trainloader, validateloader = get_train_valid_loader(path, 32, augment=True, random_seed=87, data=dataset)

oneshot_model.cuda()
oneshot_model.train()

losses = utils.AverageMeter()
top1 = utils.AverageMeter()
top5 = utils.AverageMeter()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(oneshot_model.parameters(), lr=0.0000001, momentum=0.9, weight_decay=0.0001)

for i, (inputs, labels) in enumerate(trainloader):
    inputs = inputs.cuda()
    labels = labels.cuda()

    optimizer.zero_grad()
    outputs = oneshot_model(inputs, [0]*16)

    loss = criterion(outputs, labels)
    loss.backward()

    optimizer.step()

    prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
    n = inputs.size(0)
    losses.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    
    print(top1.avg, top5.avg, losses.avg)
    if i == 100:
        break

print("------------------------------------------------------------------------")

oneshot_model.cuda()
oneshot_model.eval()

losses = utils.AverageMeter()
top1 = utils.AverageMeter()
top5 = utils.AverageMeter()
criterion = nn.CrossEntropyLoss()
    
for i, (inputs, labels) in enumerate(testloader):
    inputs = inputs.cuda()
    labels = labels.cuda()

    outputs = oneshot_model(inputs, [0]*16)

    loss = criterion(outputs, labels)

    prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
    n = inputs.size(0)
    losses.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    print(top1.avg, top5.avg, losses.avg)

torch.save(oneshot_model.state_dict(), "resnet50_oneshot.pth")