import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import random
import argparse
import utils
from models.resnet_oneshot_cifar import resnet164_oneshot
from data_loader import get_train_valid_loader, get_test_loader
from torch.nn.parallel import DistributedDataParallel as DDP

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default = "cifar100", help="cifar10/cifar100/imagenet", type=str)
    parser.add_argument("--grow", default = 2, help="number of grow supernet", type=int)
    parser.add_argument("--batch_size", default = 256, help="train batch size", type=int)
    parser.add_argument("--population", default= 20, help="Numbers of EA population", type=int)
    parser.add_argument("--search_epoch", default= 1, help="Numbers of EA search epoch", type=int)
    parser.add_argument("--topk_num", default= 10, help="Numbers of topk", type=int)
    parser.add_argument("--FLOPs", default= None, help="FLOPs constraint", type=int)
    parser.add_argument("--params", default= None, help="Numbers of parameters constraint", type=int)
    parser.add_argument("--seed", default = 87, help="random seed", type=int)
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

def check_constrain(model, arch, args):
    inputs = torch.randn((1,3,32,32))
    params, flops = utils.get_params_flops(model, inputs, arch)
    if args.params != None and args.params <= params:
        return False
    if args.FLOPs != None and args.FLOPs <= flops:
        return False
    return True

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

def validate(validate_loader, model, criterion, arch):
    model.eval()

    with torch.no_grad():
        rank = dist.get_rank()
        counter = torch.zeros((3), device=torch.device(f'cuda:{rank}'))
        for i, (inputs, labels) in enumerate(validate_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs, arch)

            loss = criterion(outputs, labels)

            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            n = inputs.size(0)
            counter[0] += prec1.item()*n
            counter[1] += prec5.item()*n
            counter[2] += n
    return counter

def search(args, model, archlist, validate_loader, criterion, backup_model, rank):
    # init EA get random population (with constrain)
    populist = []
    topk = []
    count = [0]
    if dist.get_rank() == 0:
        while count[0] < args.population:
            arch = random_model(archlist)
            isvalid = check_constrain(backup_model, arch, args)
            if isvalid == True:
                count[0]+=1
                populist.append(arch)
    dist.broadcast_object_list(count, src = 0)
    if dist.get_rank() != 0:
        populist = [None for _ in range(count[0])]
    dist.broadcast_object_list(populist, src = 0)

    n = int(len(populist) / 2)
    m = int(len(populist) / 2)
    prob = 0.1
    for i in range(args.search_epoch):
        #inference
        for p in populist:
            dist.barrier()
            print("arch:", p)
            counter = validate(validate_loader, model, criterion, p)
            dist.reduce(counter, 0)
            acc1 = counter[0].item() / counter[2].item()
            acc5 = counter[1].item() / counter[2].item()
            if dist.get_rank()==0:
                #print(rank, counter)
                print("acc1:", acc1, "acc5:", acc5)
                topk.append((p, acc1))
        
        if dist.get_rank()==0:
            #update topk
            topk = list(set(topk))
            topk.sort(key=lambda s:s[1], reverse=True)
            if len(topk) >= args.topk_num:
                topk = topk[:args.topk_num]
            print("topk:",topk)
        else:
            topk = [None for _ in range(args.topk_num)]
        dist.broadcast_object_list(topk, src = 0)
        #print(rank, topk)

        if dist.get_rank()==0:
            #crossover
            crossover_child = []
            max_iter = n * 10
            while max_iter > 0 and len(crossover_child) < n:
                max_iter-=1
                s1, s2 = random.sample(topk, 2)
                p1, _ = s1
                p2, _ = s2
                child = []
                for i,j in zip(p1,p2):
                    if isinstance(i, tuple):
                        t = []
                        for idx in range(len(i)):
                            t.append(random.choice([i[idx],j[idx]]))
                        child.append(tuple(t))
                    else:
                        child.append(random.choice([i,j]))
                if check_constrain(backup_model, child, args):
                    crossover_child.append(tuple(child))
            print("crossover done")
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
                        if isinstance(p1[i], tuple):
                            t = []
                            for j in range(len(p1[i])):
                                t.append(random.randint(0, archlist[i][j]-1))
                            p2.append(tuple(t))
                        else:
                            p2.append(random.randint(0, archlist[i]-1))
                    else:
                        p2.append(p1[i])
                if check_constrain(backup_model, p2, args):
                    mutation_child.append(tuple(p2))
            print("mutation done")
            #union
            populist = []
            populist = crossover_child + mutation_child
            count[0] = len(populist)
            print("next generation child:", count[0])

        dist.broadcast_object_list(count, src = 0)
        if dist.get_rank() != 0:
            populist = [None for _ in range(count[0])]
        dist.broadcast_object_list(populist, src = 0)

        n = int(len(populist) / 2)
        m = int(len(populist) / 2)
    return topk

def main(rank, world_size):
    setup(rank, world_size)
    args = get_args()
    model = resnet164_oneshot()
    backup_model = resnet164_oneshot()
    for i in range(args.grow):
        model.grow()
        backup_model.grow()
    archlist = model.get_all_arch()
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load("resnet164_supernet.pth"))
    model = model.cuda()
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    path = "./data/" + args.dataset
    trainloader, validateloader = get_train_valid_loader(path, args.batch_size, augment=True, random_seed=args.seed, data=args.dataset)
    topk = search(args, model, archlist, validateloader, criterion, backup_model, rank)
    if rank==0:
        file = open('./topk.txt','w+')
        with open('./topk.txt','w+') as file:
            for arch, acc in topk:
                print(arch, file=file)
    cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"You have {n_gpus} GPUs.")
    run_demo(main, n_gpus)