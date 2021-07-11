import torch
import argparse
from SuperNetEA import SuperNetEA

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default = "cifar100", help="cifar10/cifar100/imagenet", type=str)
    parser.add_argument("--arch", default = "resnet164bn_cifar100", type=str)
    parser.add_argument("--train_batch_size", default = 128, help="train batch size", type=int)
    parser.add_argument("--validate_batch_size", default = 128, help="validate batch size", type=int)
    parser.add_argument("--FLOPs", default= None, help="FLOPs constraint", type=int)
    parser.add_argument("--params", default= None, help="Numbers of parameters constraint", type=int)
    parser.add_argument("--population", default= 50, help="Numbers of EA population", type=int)
    parser.add_argument("--search_epoch", default= 20, help="Numbers of EA search epoch", type=int)
    parser.add_argument("--topk_num", default= 10, help="Numbers of topk", type=int)
    parser.add_argument("--num_workers", default= 16, help="dataloader num_workers", type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    nas = SuperNetEA(args)
    nas.load_model()
    nas.get_dataloader()
    nas.init_supernet()
    nas.grow_supernet()
    nas.grow_supernet()
    nas.pretrained_to_all_supernet()

    nas.train_supernet(100)
    nas.search()
    nas.genome_build_model(nas.select_from_topk())
    nas.print_genome()
    nas.fine_tune(lr=0.001, ftepoch=50, testepoch=1)

if __name__ == "__main__":
    main()