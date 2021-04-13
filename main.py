import torch
import argparse
from warmup import warmuper
from oneshot import OneShot

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default = "cifar100", help="cifar10/cifar100/imagenet", type=str)
    parser.add_argument("--arch", default = "resnet56", help="resnet56", type=str)
    parser.add_argument("--train_batch_size", default = 256, help="train batch size", type=int)
    parser.add_argument("--validate_batch_size", default = 256, help="validate batch size", type=int)
    parser.add_argument("--epoch", default = 10, help="number of epoch", type=int)
    parser.add_argument("--warmup_epoch", default = 10, help="number of warmup epoch", type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    warmer = warmuper(args)
    warmer.get_dataloader()
    warmer.random_group_train()
    warmer.remove_mask()
    warmer.save_model("./warmup.pth")
    spos = OneShot(args)
    spos.load_model()
    spos.build_oneshot()


if __name__ == "__main__":
    main()