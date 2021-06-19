import torch
from functools import reduce
import random
import numpy as np

def get_criterion(W):
    """ Compute the magnitude-based criterion on W.

        Returns:
            A 2d torch.Tensor of (F, C)
    """
    assert isinstance(W, torch.Tensor)
    assert W.dim() == 4

    kernel_dims = (2, 3)
    C = torch.norm(W, dim=kernel_dims)

    return C

def group_sort(C, G, num_iters=1, min_g=0):
    """ Sort the criterion matrix by the last group of
    rows and columns.

    The whole recursion works like this:
    --> sort by the last group of columns and rows, collect
        their permutation.
    --> pass the sorted C[:C.shape[0]-g_out, :C.shape[1]-g_in]
        (= C') into the next step.
    --> the returned result will be a sorted C', the
        permutation of this submatrix's indices
    --> update the matrix and indices to be returned

  Args:
  """
    assert isinstance(C, np.ndarray)

    c_out, c_in = C.shape
    g_out, g_in = c_out // G, c_in // G

    gnd_in, gnd_out = np.arange(c_in), np.arange(c_out)

    for g in reversed(range(G)):
        if g < min_g:
            break

        # heuristic method
        for _ in range(num_iters):

            # first sort the columns by the sum of the last row group
            r_lo, r_hi = g_out * g, g_out * (g + 1)
            c_lo, c_hi = g_in * g, g_in * (g + 1)

            # get the current sorting result
            # C will be updated every time
            C_ = C[gnd_out, :][:, gnd_in]

            # crop the matrix
            C_ = C_[:r_hi, :c_hi]
            # print(C_)

            # rows and cols for sorting
            rows = C_[r_lo:, :]
            perm_cols = np.argsort(rows.sum(axis=0))

            cols = C_[:, perm_cols][:, c_lo:]
            perm_rows = np.argsort(cols.sum(axis=1))

            # print(rows, rows.sum(axis=0))
            # print(cols, cols.sum(axis=1))
            # print(perm_rows, perm_cols)

            # update gnd_in and gnd_out
            gnd_in[:c_hi] = gnd_in[:c_hi][perm_cols]
            gnd_out[:r_hi] = gnd_out[:r_hi][perm_rows]

    return gnd_in, gnd_out

def get_permute_weight(w, G, num_iters):
    c = get_criterion(w)
    c /= torch.norm(c)
    gnd_in, gnd_out = group_sort(c.numpy(), G, num_iters)
    return w[gnd_out, :][:, gnd_in]

def factors(n):
    """
    Copied from https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
    """    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def get_groups_choice_list(inchannel, outchannel):
    fin, fout = factors(inchannel), factors(outchannel)
    g_list = list(sorted(fin.intersection(fout)))
    return g_list

def get_random_groups(inchannel, outchannel):
    g_list = get_groups_choice_list(inchannel, outchannel)
    return random.choice(g_list)

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

def train(train_loader, model, optimizer, scheduler, criterion, epoch):
    for e in range(epoch):
        acc1, acc5, loss = train_one_epoch(train_loader, model, optimizer, criterion)
        print("epoch", e+1, ":")
        print("top1 acc:", acc1)
        print("top5 acc:", acc5)
        print("avg loss:", loss)
        print("-------------------------------------------------")
        scheduler.step()

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