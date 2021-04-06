import torch
import torchvision
import torch.optim as optim
import utils
from models.resnet import *
from group_pruner import gconv_prune_init



class warmuper:
    def __init__(self, args):
        self.model = cifar_resnet56(pretrained=True)
        self.dataset = args.dataset
        self.train_batch_size = args.train_batch_size
        self.validate_batch_size = args.validate_batch_size
        self.epoch = args.epoch
        self.genome_type=[]

        self.init_mask()

    def create_model(self):
        pass

    def get_dataloader(self):
        if self.dataset == "cifar10":
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            self.train_set = torchvision.datasets.CIFAR10(root='./data/cifar10',train=True,transform=train_transforms,download=True)
            self.test_set = torchvision.datasets.CIFAR10(root='./data/cifar10',train=False,transform=test_transforms,download=True)
        elif self.dataset == "cifar100":
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
            test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
            self.train_set = torchvision.datasets.CIFAR100(root='./data/cifar100',train=True,transform=train_transforms,download=True)
            self.test_set = torchvision.datasets.CIFAR100(root='./data/cifar100',train=False,transform=test_transforms,download=True)
        elif self.dataset == "imagenet":
            pass

        self.trainloader = torch.utils.data.DataLoader(self.train_set, batch_size=self.train_batch_size,shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_set, batch_size=self.validate_batch_size,shuffle=False)

    def random_group_train(self):
        for i in range(10):
            print("random group train iteration:", i)
            self.set_prune_buffer_random_group()
            print(self.genome_type)
            self.train()
            self.validate()
            self.set_prune_buffer_one()

    def set_prune_buffer_one(self):
        for name, buf in self.model.named_buffers():
            if buf.dim()==4:
                buf.fill_(1)
    
    def set_prune_buffer_random_group(self):
        self.genome_type=[]
        for name, buf in self.model.named_buffers():
            if buf.dim()==4:
                n,c,h,w = buf.shape
                g = utils.get_random_groups(c, n)
                outc_start, intc_start = 0, 0
                outc_interval = int(n/g)
                intc_interval = int(c/g)
                buf.fill_(0)
                for i in range(g):
                    buf[outc_start:outc_start+outc_interval, intc_start:intc_start+intc_interval,:,:] = 1
                    outc_start+=outc_interval
                    intc_start+=intc_interval
                self.genome_type.append(g)

    def init_mask(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                gconv_prune_init(mod, name='weight')

    def validate(self):
        criterion = nn.CrossEntropyLoss()
        acc1, acc5, loss = utils.validate(self.testloader, self.model, criterion)
        print("validate:")
        print("top1 acc:", acc1)
        print("top5 acc:", acc5)    
        print("avg loss:", loss)
        print("-------------------------------------------------")

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        utils.train(self.trainloader, self.model, optimizer, criterion, self.epoch)
    
    def save_model(self, PATH):
        torch.save(self.model.state_dict(), PATH)