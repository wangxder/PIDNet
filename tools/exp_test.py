from datetime import time

import torch
from numpy import average
from torch.cuda import amp

from models.pidnet import get_seg_model, MyPIDNet, PIDNet


def calculate_flops():
    from fvcore.nn import FlopCountAnalysis, flop_count_table,ActivationCountAnalysis
    model= MyPIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128,augment=True).eval()
    x=torch.randn(1,3,1024,2048)
    flops = FlopCountAnalysis(model, x)
    print("mypid=======================================\n" + flop_count_table(flops))

    model = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128, augment=True).eval()
    x = torch.randn(1, 3, 1024, 2048)
    flops = FlopCountAnalysis(model, x)
    print("pid=======================================\n" + flop_count_table(flops))

def calculate_params(model):
    #https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    model_parameters = model.parameters()
    params2 = sum([np.prod(p.size()) for p in model_parameters])
    return params,params2

def cityscapes_speed_test():
    print("cityscapes speed test")
    model=MyPIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128,augment=True).eval()
    x=torch.randn(1,3,1024,2048)
    ts=[]
    ts.extend(benchmark_eval([model],x,True))
    print(ts)

def camvid_speed_test():
    print("camvid speed test")
    model=MyPIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128,augment=True).eval()
    x=torch.randn(1,3,720,960)
    ts=[]
    ts.extend(benchmark_eval([model],x,True))
    print(ts)

def benchmark_eval(models,x,mixed_precision):
    torch.backends.cudnn.benchmark=True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x=x.to(device)
    ts=[]
    for model in models:
        model=model.to(device)
        t=compute_eval_time2(model,x,10,100,mixed_precision)
        model.cpu()
        print(t)
        ts.append(t)
    return ts

@torch.no_grad()
def compute_eval_time2(model,x,warmup_iter,num_iter,mixed_precision):
    model.eval()
    times=[]
    for cur_iter in range(warmup_iter+num_iter):
        if cur_iter == warmup_iter:
            times.clear()
        t1=time.time()
        with amp.autocast(enabled=mixed_precision):
            output = model(x)
        torch.cuda.synchronize()
        t2=time.time()
        times.append(t2-t1)
    return average(times)

if __name__ == "__main__":
    calculate_flops()