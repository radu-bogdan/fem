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
gmsh.option.setNumber("Mesh.MeshSizeMax", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", 1)

p,e,t,q = pde.petq_generate()
MESH = pde.mesh(p,e,t,q)
MESH.makeFemLists(space = 'SP1')



B00,B01,B10,B11 = pde.hdivdiv.assemble(MESH, space = 'SP1', matrix = 'M', order = 1)
D2 = pde.int.assemble(MESH, order = 1)

K = B00@D2@B00.T + B10@D2@B10.T + B01@D2@B01.T + B11@D2@B11.T

A = cholesky(K)


N = A.L()@A.L().T


# def fast_inverse(A):
#     identity = np.identity(A.shape[2], dtype=A.dtype)
#     Ainv = np.zeros_like(A)

#     for i in range(A.shape[0]):
#         Ainv[i] = np.linalg.solve(A[i], identity)
#     return Ainv

# @profile

# def invert_block(N):
        
#####################################################################################
# Find indices where the blocks begin/end
#####################################################################################
tm = time.time()

N_diag = N.diagonal(k=-1) # Nebendiagonale anfangen
block_ends = np.r_[np.argwhere(abs(N_diag)==0)[:,0],N.shape[0]-1]

for i in range(N.shape[0]):
    N_diag = np.r_[N.diagonal(k=-(i+2)),np.zeros(i+2)]
    
    for j in range(i+1):
        arg = np.argwhere(abs(N_diag[block_ends-j])>0)[:,0]
        block_ends = np.delete(block_ends,arg).copy()
        
    if np.linalg.norm(N_diag)==0: break

block_ends = np.r_[-1,block_ends]
block_lengths = np.r_[block_ends[1:]-block_ends[0:-1]]

elapsed = time.time()-tm; print('Preparing lists {:4.8f} seconds.'.format(elapsed))
#####################################################################################


#####################################################################################
# Inversion of the blocks:
#####################################################################################
iN = sps.lil_matrix(N.shape)

tm = time.time()
for i in range(len(block_ends)-1):
    
    C = N[block_ends[i]+1:block_ends[i+1]+1,
          block_ends[i]+1:block_ends[i+1]+1].toarray()
    
    iC = np.linalg.inv(C)
    
    iN[block_ends[i]+1:block_ends[i+1]+1,
       block_ends[i]+1:block_ends[i+1]+1] = iC
    
elapsed = time.time()-tm; print('Inverting took {:4.8f} seconds.'.format(elapsed))
#####################################################################################


#####################################################################################
# Inversion of the blocks, 2nd try.
#####################################################################################

dataN = N.data
indicesN = N.indices
indptrN = N.indptr    
lengths = N.indptr[1:]-N.indptr[0:-1]

print(indicesN)
print(indptrN)


# ind = np.empty(0,dtype=int)
# for i,val in enumerate(leng):
#     ind = np.append(ind,np.tile(i,val))

# print(ind)
# print(ind.shape)
print(indicesN.shape)
print('dasdasd')

pp = 0

block_ends = block_ends + 1

for i,ii in enumerate(block_ends):
    
    print(i,ii)
    
    
    CC = np.zeros(shape=(block_lengths[i],block_lengths[i]))
    
    for k in range(ii):
        # print(k)
        print(indicesN[indptrN[k]:indptrN[k+1]])
        CC[k-ii,indicesN[indptrN[k]:indptrN[k+1]]] = dataN[indptrN[k]:indptrN[k+1]]
    
    
    # for k in 
    
    # pos = np.argwhere((N.indices>block_ends[i]) & (N.indices<=block_ends[i+1]))[:,0]
    # indices = indicesN[pos]
    
    # C = dataN[np.argwhere((N.indices>block_ends[i]) & (N.indices<=block_ends[i+1]))]
    
    # CC = np.zeros(shape=(block_lengths[i],block_lengths[i]))
    
    # print(lengths)
        
    # for i in range()
    #     CC[] = dataN[k*(j+1)]
    
    
    print(CC)
    # print(indices)
#####################################################################################

# return iN






# iN = invert_block(N)

# tm = time.time()
# iN2 = sps.linalg.inv(N)
# elapsed = time.time()-tm; print('Inverting in the classic way took {:4.8f} seconds.'.format(elapsed))
    
# print(sps.linalg.norm(iN2-iN))

print(N.shape)

m = 1
plt.close('all')
plt.figure(); spy(N,markersize = m)
plt.figure(); spy(iN,markersize = m)

import numpy as np

mat = np.zeros((5, 5))
indexes = np.array([[0, 2], [1, 3], [1, 4], [0, 2], [1, 4]])
values = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

for row_i, column_indices in enumerate(indexes):
    print(row_i)
    print(column_indices)
    mat[row_i, column_indices] = values[row_i]



# plt.figure(); spy(N,markersize = m)
# plt.figure(); spy(iN-iN2,precision=1e-12,markersize = m)



# print(iN2[41:41+12,41:41+12].toarray(),'\n\n')
# print(N[41:41+12,41:41+12].toarray())