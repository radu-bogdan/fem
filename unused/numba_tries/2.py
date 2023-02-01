import numba as nb
import numpy as np
import time

# @nb.jit(parallel=True,cache=True)
@nb.jit(cache=True)
def stacking(a,b):
    return np.hstack((a,b))

def stacking2(a,b):
    return np.hstack((a,b))

zeilen = 100000
spalten = 1000

a = np.random.rand(zeilen,spalten)
b = np.random.rand(zeilen,spalten)

tm = time.time()
x = stacking(a,b)
print(time.time()-tm)

tm = time.time()
x = stacking2(a,b)
print(time.time()-tm)

# tm = time.time()
# ident_loops(x)
# print(time.time()-tm)