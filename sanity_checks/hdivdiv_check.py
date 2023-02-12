
import sys
sys.path.insert(0,'..') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
import numba as nb
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import geometries
from sksparse.cholmod import cholesky

import plotly.io as pio
pio.renderers.default = 'browser'
from matplotlib.pyplot import spy
import matplotlib.pyplot as plt



np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = np.inf)
np.set_printoptions(precision = 2)

gmsh.initialize()
gmsh.model.add("Capacitor plates")
geometries.unitSquare()
gmsh.option.setNumber("Mesh.Algorithm", 2)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 1)
# gmsh.option.setNumber("Mesh.MeshSizeMin", 1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.0017)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0017)

p,e,t,q = pde.petq_generate()
MESH = pde.mesh(p,e,t,q)
MESH.makeFemLists(space = 'SP1')



tm = time.time()
B00,B01,B10,B11 = pde.hdivdiv.assemble(MESH, space = 'SP1', matrix = 'M', order = 1)
D2 = pde.int.assemble(MESH, order = 1)

K = B00@D2@B00.T + B10@D2@B10.T + B01@D2@B01.T + B11@D2@B11.T
elapsed = time.time()-tm; print('Assembling took {:4.8f} seconds.'.format(elapsed))

A = cholesky(K)


N = A.L()@A.L().T


# def fast_inverse(A):
#     identity = np.identity(A.shape[2], dtype=A.dtype)
#     Ainv = np.zeros_like(A)

#     for i in range(A.shape[0]):
#         Ainv[i] = np.linalg.solve(A[i], identity)
#     return Ainv

# @profile



# tm = time.time()
# data_iN,indices_iN,indptr_iN = whatever(dataN,indicesN,indptrN,block_ends)
# elapsed = time.time()-tm; print('Njit took {:4.8f} seconds.'.format(elapsed))

# tm = time.time()
# iN = sps.csc_matrix((data_iN, indices_iN, indptr_iN), shape = N.shape)
# elapsed = time.time()-tm; print('csc took {:4.8f} seconds.'.format(elapsed))
#####################################################################################

iN = pde.tools.fastBlockInverse(N)


# mysum.inspect_llvm()
# mysum.inspect_asm()



# print(indptr_iN)
# print(indices_iN)



# iN,iN2 = invert_block(N)

# tm = time.time()
# iN2 = sps.linalg.inv(N)
# elapsed = time.time()-tm; print('Inverting in the classic way took {:4.8f} seconds.'.format(elapsed))
    
# print('kekw', sps.linalg.norm(iN2-iN))

print(N.shape)

# m = 1
# plt.close('all')
# plt.figure(); spy(N,markersize = m)
# plt.figure(); spy(iN,markersize = m)

Id = N@iN
print(sps.linalg.norm(N@iN,np.inf))





# plt.figure(); spy(N,markersize = m)
# plt.figure(); spy(iN-iN2,precision=1e-12,markersize = m)
