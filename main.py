import torch
import argparse
from SuperNetEA import SuperNetEA

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default = "cifar100", help="cifar10/cifar100/imagenet", type=str)
    parser.add_argument("--arch", default = "resnet164bn_cifar100", type=str)
    parser.add_argument("--train_batch_size", default = 128, help="train batch size", type=int)
    parser.add_argument("--validate_batch_size", default = 128, help="validate batch size", type=int)
    parser.add_argument("--epoch", default = 10, help="number of epoch", type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    nas = SuperNetEA(args)
    nas.load_model()
    nas.get_dataloader()
    nas.build_supernet()
    nas.pretrained_to_supernet()

    for i in range(100):
        print("number:", i+1)
        nas.random_model_train_validate()

    for i in range(15):
        print("number:", i+1)
        nas.random_model()
        nas.print_genome()
        nas.count_flops_params()
        nas.validate()

    nas.random_model()
    nas.print_genome()
    nas.count_flops_params()
    nas.fine_tune()
    nas.validate()


if __name__ == "__main__":
    main()