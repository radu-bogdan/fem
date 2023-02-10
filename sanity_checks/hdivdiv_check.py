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
gmsh.option.setNumber("Mesh.MeshSizeMax", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", 1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.002)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.002)

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

def invert_block(N):
        
    N = N.tocsc()
    
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
        
    elapsed = time.time()-tm; print('Inverting1 took {:4.8f} seconds.'.format(elapsed))
    #####################################################################################
    
    iN2 = 0
    
    #####################################################################################
    # Inversion of the blocks, 2nd try.
    #####################################################################################
    # tm = time.time(); A = ()
    # for i in range(len(block_ends)-1):
    #     C = N[block_ends[i]+1:block_ends[i+1]+1,
    #           block_ends[i]+1:block_ends[i+1]+1].toarray()
        
    #     iC = np.linalg.inv(C)
        
    #     A += (iC,)
    
    # from scipy.sparse import block_diag as bd_sp
    # iN2 = bd_sp(A)
    # elapsed = time.time()-tm; print('Inverting2 took {:4.8f} seconds.'.format(elapsed))
    #####################################################################################

    return iN,iN2





iN,iN2 = invert_block(N)

# tm = time.time()
# iN2 = sps.linalg.inv(N)
# elapsed = time.time()-tm; print('Inverting in the classic way took {:4.8f} seconds.'.format(elapsed))
    
print('kekw', sps.linalg.norm(iN2-iN))

print(N.shape)

m = 1
plt.close('all')
plt.figure(); spy(N,markersize = m)
plt.figure(); spy(iN,markersize = m)



# plt.figure(); spy(N,markersize = m)
# plt.figure(); spy(iN-iN2,precision=1e-12,markersize = m)



# print(iN2[41:41+12,41:41+12].toarray(),'\n\n')
# print(N[41:41+12,41:41+12].toarray())








def numbaspeed(dataN,indicesN,indptrN,block_ends,block_lengths):
    block_ends = block_ends + 1
    A = ()
    
    for i,ii in enumerate(block_ends[:-1]):
        
        CC = np.zeros(shape=(block_lengths[i],block_lengths[i]),dtype=np.float64)
        
        for k in range(block_lengths[i]):
            in_k = np.empty(shape=(0,),dtype=np.int64)
            
            for el in range(indptrN[block_ends[i]+k+1]-indptrN[block_ends[i]+k]):
                in_k = np.append(in_k,indptrN[block_ends[i]+k]+el)
            
            # in_k = np.r_[indptrN[block_ends[i]+k]:indptrN[block_ends[i]+k+1]]
            # print(in_k)
            
            # print(k,indicesN[in_k]-block_ends[i],dataN[in_k])
            # print(CC[k,:])
            # print(CC[:,indicesN[in_k]-block_ends[i]])
            
            # CCk = CC[k,:]
            # CCk[indicesN[in_k]-block_ends[i]] = dataN[in_k]
            # CC[k,:] = CCk
            
            
            CC[k,indicesN[in_k]-block_ends[i]] = dataN[in_k]
            
            
        iCC = np.linalg.inv(CC)
        A += (iCC,)
    return A