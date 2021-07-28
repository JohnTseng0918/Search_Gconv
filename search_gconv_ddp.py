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
from data_loader_ddp import get_train_valid_loader, get_test_loader
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

def train(model, args, trainloader, archlist, criterion, rank):
    model.train()
    model = model.to(rank)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0001, momentum=0.9)
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

        print("epoch:", e+1)
        print("top1 acc:", top1.avg)
        print("top5 acc:", top5.avg)
        print("avg loss:", losses.avg)
        print("-------------------------------------------------")

def main(rank, world_size):
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

    print(f"Running main(**args) on rank {rank}.")
    setup(rank, world_size)

    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)


    trainloader, validateloader = get_train_valid_loader("./data/cifar100", args.batch_size, augment=True, random_seed=87, data=args.dataset)
    testloader = get_test_loader("./data/cifar100", args.batch_size, shuffle=False, data=args.dataset)

    train(ddp_model, args, trainloader, archlist, criterion, rank)
    print("train model done")
    if rank==0:
        torch.save(ddp_model.state_dict(), "./resnet164_supernet.py")
    cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"You have {n_gpus} GPUs.")
    run_demo(main, n_gpus)