import torch

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def evaluation(outputs, labels):
    correct = torch.sum(torch.eq(torch.argmax(outputs, dim=1), labels)).item()
    return correct

def validate(validate_loader, model, criterion):
    model.cuda()
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    
    for i, (inputs, labels) in enumerate(validate_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        correct = evaluation(outputs, labels)

        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
        n = inputs.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    return top1.avg, top5.avg, losses.avg

def train_one_epoch(train_loader, model, optimizer, criterion):
    model.cuda()
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
        n = inputs.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    return top1.avg, top5.avg, losses.avg

def train(train_loader, model, optimizer, criterion, epoch):
    for e in range(epoch):
        acc1, acc5, loss = train_one_epoch(train_loader, model, optimizer, criterion)
        print("epoch", e+1, ":")
        print("top1 acc:", acc1)
        print("top5 acc:", acc5)
        print("avg loss:", loss)
        print("-------------------------------------------------")

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L363-L384
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)