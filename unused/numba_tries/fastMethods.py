import numba as nb
import numpy as np

@nb.njit(cache=True)
# @nb.jit(cache=True)
def stacking(a,b):
    return np.vstack((a,b))

def stacking2(a,b):
    # return np.c_[a,b]
    return np.vstack((a,b))