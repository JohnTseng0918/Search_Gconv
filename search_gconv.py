import torch
import torch.nn as nn
import argparse
import random
import utils
import torch.optim as optim
from models.resnet_oneshot_cifar import resnet164_oneshot
from data_loader import get_train_valid_loader, get_test_loader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default = "cifar100", help="cifar10/cifar100/imagenet", type=str)
    parser.add_argument("--epoch", default = 1, help="numbers of train supernet epoch", type=int)
    parser.add_argument("--lr", default = 0.05, help="train supernet learning rate", type=float)
    parser.add_argument("--batch_size", default = 128, help="train batch size", type=int)
    parser.add_argument("--population", default= 20, help="Numbers of EA population", type=int)
    parser.add_argument("--search_epoch", default= 10, help="Numbers of EA search epoch", type=int)
    parser.add_argument("--topk_num", default= 10, help="Numbers of topk", type=int)
    parser.add_argument("--FLOPs", default= None, help="FLOPs constraint", type=int)
    parser.add_argument("--params", default= None, help="Numbers of parameters constraint", type=int)
    parser.add_argument("--ftepoch", default = 10, help="numbers of fine tune epoch", type=int)
    parser.add_argument("--ftlr", default = 0.005, help="fine tune learning rate", type=float)
    args = parser.parse_args()
    return args

def random_model(archlist):
    retlist = []
    for i in archlist:
        retlist.append(random.randint(0, i-1))
    return tuple(retlist)

def train(model, args, trainloader, archlist, criterion):
    model.train()
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    for e in range(args.epoch):
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        for inputs, labels in trainloader:
            arch = random_model(archlist)

            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs, arch)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            n = inputs.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
        
        scheduler.step()
        
        print("epoch:", e+1)
        print("top1 acc:", top1.avg)
        print("top5 acc:", top5.avg)
        print("avg loss:", losses.avg)
        print("-------------------------------------------------")

def validate(validate_loader, model, criterion, arch):
    model.eval()

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validate_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs, arch)

            loss = criterion(outputs, labels)

            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            n = inputs.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    return top1.avg, top5.avg, losses.avg

def search(args, model, archlist, validate_loader, criterion, backup_model):
    # init EA get random population (with constrain)
    populist = []
    topk = []
    count=0
    while count < args.population:
        arch = random_model(archlist)
        isvalid = check_constrain(backup_model, arch, args)
        if isvalid == True:
            count+=1
            populist.append(arch)
    
    n = int(len(populist) / 2)
    m = int(len(populist) / 2)
    prob = 0.1
    for i in range(args.search_epoch):
        #inference
        for p in populist:
            print(p)
            acc1, acc5, loss = validate(validate_loader, model, criterion, p)
            print("acc1:", acc1)
            print("acc5:", acc5)
            print("loss:", loss)
            print("-------------------------------------------------")
            topk.append((p,acc1))
        
        #update topk
        topk = list(set(topk))
        topk.sort(key=lambda s:s[1], reverse=True)
        if len(topk) >= args.topk_num:
            topk = topk[:args.topk_num]
        print("topk:",topk)

        #crossover
        crossover_child = []
        max_iter = n * 10
        while max_iter > 0 and len(crossover_child) < n:
            max_iter-=1
            s1, s2 = random.sample(topk, 2)
            p1, _ = s1
            p2, _ = s2
            child = [random.choice([i,j]) for i,j in zip(p1,p2)]
            if check_constrain(backup_model, child, args):
                crossover_child.append(tuple(child))
        

        #mutation
        mutation_child = []
        max_iter = m * 10
        while max_iter > 0 and len(mutation_child) < m:
            max_iter-=1
            s1 = random.choice(topk)
            p1, _ = s1
            p2 = []
            for i in range(len(p1)):
                if random.random() < prob:
                    p2.append(random.randint(0, archlist[i]-1))
                else:
                    p2.append(p1[i])
            if check_constrain(backup_model, p2, args):
                mutation_child.append(tuple(p2))

        #union
        populist = []
        populist = crossover_child + mutation_child
        n = int(len(populist) / 2)
        m = int(len(populist) / 2)
    return topk

def check_constrain(model, arch, args):
    inputs = torch.randn((1,3,32,32))
    params, flops = utils.get_params_flops(model, inputs, arch)
    if args.params != None and args.params <= params:
        return False
    if args.FLOPs != None and args.FLOPs <= flops:
        return False
    return True

def fine_tune(model, args, trainloader, arch, criterion):
    model.train()
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.ftlr, weight_decay=0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ftepoch)
    for e in range(args.ftepoch):
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        for inputs, labels in trainloader:

            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs, arch)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            n = inputs.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
        
        scheduler.step()
        
        print("fine tune epoch:", e+1)
        print("top1 acc:", top1.avg)
        print("top5 acc:", top5.avg)
        print("avg loss:", losses.avg)
        print("-------------------------------------------------")

def main():
    args = get_args()
    model = resnet164_oneshot()
    model.load_state_dict(torch.load("./pretrained/resnet164_cifar100_oneshot.pth"))
    model.grow_with_pretrained()
    model.grow_with_pretrained()
    archlist = model.get_all_arch()
    criterion = nn.CrossEntropyLoss()

    backup_model = resnet164_oneshot()
    backup_model.grow()
    backup_model.grow()

    trainloader, validateloader = get_train_valid_loader("./data/cifar100", args.batch_size, augment=True, random_seed=87, data=args.dataset)
    testloader = get_test_loader("./data/cifar100", args.batch_size, shuffle=False, data=args.dataset)

    train(model, args, trainloader, archlist, criterion)
    topk = search(args, model, archlist, validateloader, criterion, backup_model)
    arch, _ = topk[0]
    fine_tune(model, args, trainloader, arch, criterion)

if __name__ == "__main__":
    main()