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
from data_loader import get_train_valid_loader, get_test_loader
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
    parser.add_argument("--epoch", default = 300, help="numbers of train supernet epoch", type=int)
    parser.add_argument("--lr", default = 0.1, help="train supernet learning rate", type=float)
    parser.add_argument("--momentum", default = 0.9, help="train supernet momentum", type=float)
    parser.add_argument("--weight_decay", default = 0.0001, help="train supernet weight_decay", type=float)
    parser.add_argument("--batch_size", default = 128, help="train batch size", type=int)
    parser.add_argument("--seed", default = 87, help="random seed", type=int)
    args = parser.parse_args()
    return args

def main(rank, world_size):
    args = get_args()
    model = resnet164_oneshot()
    criterion = nn.CrossEntropyLoss()

    print(f"Running main(**args) on rank {rank}.")
    setup(rank, world_size)

    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    path = "./data/" + args.dataset
    trainloader, validateloader = get_train_valid_loader(path, args.batch_size, augment=True, random_seed=args.seed, valid_size = 0,data=args.dataset)
    testloader = get_test_loader(path, args.batch_size, shuffle=False, data=args.dataset)
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    arch = tuple([0]*54)
    for e in range(args.epoch):
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = ddp_model(inputs, arch)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            n = inputs.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        print("train top1 acc:", top1.avg, "train top5 acc:",top5.avg, "train avg loss:", losses.avg)
        scheduler.step()

        if e > args.epoch*0.5 and rank==0:
            model.eval()
            testlosses = utils.AverageMeter()
            testtop1 = utils.AverageMeter()
            testtop5 = utils.AverageMeter()
            
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(testloader):
                    inputs = inputs.to(rank)
                    labels = labels.to(rank)

                    outputs = model(inputs, arch)

                    loss = criterion(outputs, labels)

                    prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
                    n = inputs.size(0)
                    testlosses.update(loss.item(), n)
                    testtop1.update(prec1.item(), n)
                    testtop5.update(prec5.item(), n)

            print("test top1 acc:", testtop1.avg, "test top5 acc:",testtop5.avg, "test avg loss:", testlosses.avg)
            
        
    if rank==0:
        torch.save(ddp_model.module.state_dict(), "./resnet164_supernet.pth")
    cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"You have {n_gpus} GPUs.")
    run_demo(main, n_gpus)