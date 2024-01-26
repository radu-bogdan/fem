import numpy as np
import scipy.sparse as sp

def removeZeros(A,tol=1e-13):
    indices = np.where((np.diff(A.indptr) != 0))[0]
    D = sp.eye(A.shape[0], format = 'csc')
    R = D[indices,:]
    return R