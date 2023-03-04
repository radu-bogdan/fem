import numpy as np
import torch
import time

N = 2048
a = torch.randn((N,N), dtype=torch.float32)#.to('cpu')
b = torch.randn((N,N), dtype=torch.float32)#.to('cpu')

def axb():
    print("Starting matmul")
    st = time.monotonic()
    c = a@b
    return time.monotonic()-st

flops = N*N*N*2

et = min([axb() for _ in range(10)])
print(f"{et*1e6:.2f} us, {flops*1e-9/et:.2f} GFLOPS")