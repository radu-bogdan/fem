import sys
sys.path.insert(0,'..') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
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


np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = np.inf)
np.set_printoptions(precision = 8)

gmsh.initialize()
gmsh.model.add("Capacitor plates")
geometries.unitSquare()
gmsh.option.setNumber("Mesh.Algorithm", 2)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.01)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)

p,e,t,q = pde.petq_generate()
MESH = pde.mesh(p,e,t,q)
MESH.makeFemLists(space = 'SP1')



B00,B01,B10,B11 = pde.hdivdiv.assemble(MESH, space = 'SP1', matrix = 'M', order = 1)
D2 = pde.int.assemble(MESH, order = 1)

K = B00@D2@B00.T + B10@D2@B10.T + B01@D2@B01.T + B11@D2@B11.T

A = cholesky(K)


N = A.L()@A.L().T

# spy(N)

# # block_ends = np.zeros()

# arr = np.empty(shape = (0,N.shape[0]))
# for i in range(N.shape[0]):
    
#     N_diag = N.diagonal(k=-i) # Nebendiagonale anfangen
    
#     block_ends = np.argwhere(abs(N_diag)<1e-10)[:,0]
    
#     if np.linalg.norm(N_diag)<1e-10:
#         break
    
#     values = np.r_[N.diagonal(k=-i),np.zeros(i)]
#     arr = np.r_[arr, values[None]]


# from scipy.linalg import solveh_banded
# D = scipy.sparse.eye(N.shape[0])
# invN = solveh_banded(arr, D.toarray() , lower=True)

# print(arr.shape)
# spy(invN)
# # N = A.L()@A.L().T




tm = time.time()
N_diag = N.diagonal(k=-1) # Nebendiagonale anfangen
block_ends = np.argwhere(abs(N_diag)<1e-6)[:,0]

for i in range(N.shape[0]):
    N_diag = np.r_[N.diagonal(k=-(i+2)),np.zeros(i)]
    arg = np.argwhere(abs(N_diag[block_ends])>1e-10)[:,0]
    # print(arg)
    # print(block_ends)
    block_ends = np.delete(block_ends,arg).copy()
    # print(block_ends)
    
    if np.linalg.norm(N_diag)<1e-10:
        break


elapsed = time.time()-tm; print('Solving took  {:4.8f} seconds.'.format(elapsed))


for i in range(block_ends):
    
    
    