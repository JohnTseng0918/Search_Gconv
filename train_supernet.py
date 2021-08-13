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
from models.densenet_oneshot_cifar import condensenet86_oneshot
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
    parser.add_argument("--grow", default = 2, help="number of grow supernet", type=int)
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

def main(rank, world_size):
    args = get_args()
    model = condensenet86_oneshot()
    #model.load_state_dict(torch.load("./pretrained/resnet50_oneshot.pth"))
    for i in range(args.grow):
        model.grow_with_pretrained()
    archlist = model.get_all_arch()
    criterion = nn.CrossEntropyLoss()

    print(f"Running main(**args) on rank {rank}.")
    setup(rank, world_size)

    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    path = "./data/" + args.dataset
    trainloader, validateloader = get_train_valid_loader(path, args.batch_size, augment=True, random_seed=args.seed, data=args.dataset)

    train(ddp_model, args, trainloader, archlist, criterion, rank)
    if rank==0:
        torch.save(ddp_model.module.state_dict(), "./condensenet86_supernet.pth")
    cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"You have {n_gpus} GPUs.")
    run_demo(main, n_gpus)
