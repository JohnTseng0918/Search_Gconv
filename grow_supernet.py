import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import random
import utils
import torch.optim as optim
from models.resnet_oneshot_cifar import resnet164_oneshot
from models.resnet_oneshot import resnet50_oneshot
from data_loader import get_train_valid_loader
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default = "cifar100", help="cifar10/cifar100/imagenet", type=str)
    parser.add_argument("--epoch", default = 1, help="numbers of train supernet epoch", type=int)
    parser.add_argument("--lr", default = 0.05, help="train supernet learning rate", type=float)
    parser.add_argument("--momentum", default = 0.9, help="train supernet momentum", type=float)
    parser.add_argument("--weight_decay", default = 0.0001, help="train supernet weight_decay", type=float)
    parser.add_argument("--batch_size", default = 256, help="train batch size", type=int)
    parser.add_argument("--FLOPs", default= None, help="FLOPs constraint", type=int)
    parser.add_argument("--params", default= None, help="Numbers of parameters constraint", type=int)
    parser.add_argument("--seed", default = 87, help="random seed", type=int)
    args = parser.parse_args()
    return args

def random_model(archlist):
    retlist = []
    for i in archlist:
        if isinstance(i, tuple):
            t = []
            for num in i:
                t.append(random.randint(0, num-1))
            retlist.append(tuple(t))
        else:
            retlist.append(random.randint(0, i-1))
    return tuple(retlist)

def train(model, args, trainloader, archlist, criterion, rank):
    model.train()
    model = model.to(rank)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    for e in range(args.epoch):
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        for inputs, labels in trainloader:
            arch = random_model(archlist)


            inputs = inputs.to(rank)
            labels = labels.to(rank)

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

        print("epoch:", e+1, "top1 acc:", top1.avg, "top5 acc:", top5.avg, "avg loss:", losses.avg)

def check_constrain(model, arch, args):
    inputs = torch.randn((1,3,32,32))
    params, flops = utils.get_params_flops(model, inputs, arch)
    if args.params != None and args.params <= params:
        return False
    if args.FLOPs != None and args.FLOPs <= flops:
        return False
    return True

def check_supernet_constraints(model, archlist, args):
    valid_count=0
    for i in range(100):
        arch = random_model(archlist)
        ans = check_constrain(model, arch, args)
        if ans == True:
            valid_count+=1
    print("valid_count:", valid_count)
    if valid_count >=10:
        return True
    else:
        return False

def main(rank, world_size):
    print(f"Running main(**args) on rank {rank}.")
    setup(rank, world_size)
    args = get_args()
    model = resnet164_oneshot()
    backup_model = resnet164_oneshot()
    model.load_state_dict(torch.load("resnet164_cifar100_oneshot.pth"))

    criterion = nn.CrossEntropyLoss()
    path = "./data/" + args.dataset
    trainloader, validateloader = get_train_valid_loader(path, args.batch_size, augment=True, random_seed=args.seed, data=args.dataset)
    
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    count=0

    while True:
        model.grow_with_pretrained()
        backup_model.grow()
        count+=1
        archlist = model.get_all_arch()
        train(ddp_model, args, trainloader, archlist, criterion, rank)
        flag = [None]
        if dist.get_rank() == 0:
            flag[0] = check_supernet_constraints(backup_model, archlist, args)
        dist.broadcast_object_list(flag, src = 0)
        if flag[0]==True:
            break
    print("total grow:", count)

    if rank==0:
        torch.save(ddp_model.module.state_dict(), "resnet164_supernet.pth")
    cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"You have {n_gpus} GPUs.")
    run_demo(main, n_gpus)
