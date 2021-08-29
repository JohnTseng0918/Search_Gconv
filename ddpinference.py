from pytorchcv.model_provider import get_model as ptcv_get_model
from dataloadertest import get_train_valid_loader, get_test_loader, get_test_loader
import utils
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os

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


def main(rank, world_size):
    model = ptcv_get_model("resnet164bn_cifar100", pretrained=True)
    model = model.to(rank)
    model.eval()

    validate_loader = get_test_loader("./data/cifar100", batch_size=128, shuffle=False)

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validate_loader):
            inputs = inputs.to(rank)
            labels = labels.to(rank)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            prec1, prec5 = utils.accuracy(outputs, labels, topk=(1, 5))
            n = inputs.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            print(top1.avg, top5.avg, losses.avg)
    print("-----------------------------------------------------------------------------")
    print(top1.avg, top5.avg, losses.avg)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    run_demo(main, n_gpus)