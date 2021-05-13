import torch
import argparse
from spos import SuperNet

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default = "cifar100", help="cifar10/cifar100/imagenet", type=str)
    parser.add_argument("--arch", default = "resnet56", help="resnet56", type=str)
    parser.add_argument("--train_batch_size", default = 256, help="train batch size", type=int)
    parser.add_argument("--validate_batch_size", default = 256, help="validate batch size", type=int)
    parser.add_argument("--epoch", default = 10, help="number of epoch", type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    
    nas = SuperNet(args)
    nas.load_model()
    nas.build_oneshot()
    nas.get_dataloader()
    nas.warnup_oneshot()
    for i in range(100):
        nas.train_supernet()


if __name__ == "__main__":
    main()