import numba as nb
import numpy as np
import time

@nb.jit(parallel=True,cache=True)
def ident_np(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2
# @nb.jit(parallel=True,cache=True)

def ident_np2(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2

@nb.jit(cache=True)
def ident_loops(x):
    r = np.empty_like(x)
    n = len(x)
    for i in range(n):
        r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
    return r



x = np.random.rand(1_000_000_000)

tm = time.time()
ident_np(x)
print(time.time()-tm)

tm = time.time()
ident_np2(x)
print(time.time()-tm)

# tm = time.time()
# ident_loops(x)
# print(time.time()-tm)