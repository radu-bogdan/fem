import cupy as cp
import numpy as np
import scipy.sparse as sp
from cupyx.scipy.sparse import csr_matrix as cp_csc_matrix

nA = np.array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])

nAs = sp.csc_matrix(nA, dtype = cp.float32)

# cA = cp.array(nA, dtype=cp.float32)
cA_csr = cp_csc_matrix(nAs)