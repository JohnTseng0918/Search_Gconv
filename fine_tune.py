import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import utils
from torch.nn.parallel import DistributedDataParallel as DDP
from data_loader import get_train_valid_loader, get_test_loader
from models.resnet_oneshot_cifar import resnet164_oneshot

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default = "cifar100", help="cifar10/cifar100/imagenet", type=str)
    parser.add_argument("--grow", default = 2, help="number of grow supernet", type=int)
    parser.add_argument("--batch_size", default = 256, help="train batch size", type=int)
    parser.add_argument("--seed", default = 87, help="random seed", type=int)
    parser.add_argument("--ftepoch", default = 1, help="numbers of fine tune epoch", type=int)
    parser.add_argument("--ftlr", default = 0.005, help="fine tune learning rate", type=float)
    args = parser.parse_args()
    return args

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

def fine_tune(model, args, trainloader, testloader, arch, criterion, rank):
    model = model.to(rank)
    optimizer = optim.SGD(model.parameters(), lr=args.ftlr, weight_decay=0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ftepoch)
    for e in range(args.ftepoch):
        model.train()
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        for inputs, labels in trainloader:

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
        
        print("fine tune epoch:", e+1, "top1 acc:", top1.avg, "top5 acc:", top5.avg, "avg loss:", losses.avg)

        if rank == 0:
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

            print("top1 acc:", testtop1.avg, "top5 acc:",testtop5.avg, "avg loss:", testlosses.avg)


def main(rank, world_size):
    args = get_args()
    model = resnet164_oneshot()
    for i in range(args.grow):
        model.grow()
    archlist = model.get_all_arch()
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load("./resnet164_supernet.pth"))
    with open("./topk.txt") as f:
        lines = f.readlines()
    
    topk = []
    for i in range(len(lines)):
        a = lines[i]
        a = a.replace(" ", "")
        a = a.replace("(", "")
        a = a.replace(")", "")
        a = a.replace("\n", "")
        l = a.split(",")
        l = tuple(map(int, l))
        topk.append(l)

    arch = topk[0]

    print(f"Running main(**args) on rank {rank}.")
    setup(rank, world_size)

    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)


    trainloader, validateloader = get_train_valid_loader("./data/cifar100", args.batch_size, augment=True, random_seed=args.seed, data=args.dataset)
    testloader = get_test_loader("./data/cifar100", args.batch_size, shuffle=False, data=args.dataset)

    fine_tune(ddp_model, args, trainloader, testloader, arch, criterion, rank)
    ddp_model = ddp_model.to("cpu")

    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "./resnet164_finetune_supernet.pth")
        inputs = torch.randn((1,3,32,32))
        params, flops = utils.get_params_flops(model, inputs, arch)
        print("pamras:", params, "flops:", flops)

    cleanup()
    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"You have {n_gpus} GPUs.")
    run_demo(main, n_gpus)