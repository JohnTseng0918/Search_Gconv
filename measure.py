import torch
import time
from models.resnet_oneshot import resnet50_oneshot
from models.resnet_oneshot_cifar import resnet164_oneshot

model = resnet50_oneshot()
arch = model.get_origin_arch()

model = model.cuda()
model.eval()
inputs = torch.randn(32, 3, 224, 224).cuda()

time_list = []
for i in range(70):
    torch.cuda.synchronize()
    start = time.time()
    result = model(inputs, arch)
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end-start)
avg_time = sum(time_list[20:])/len(time_list[20:])
print(avg_time)
