import sys
import time
import numpy as np
import numba as nb
import scipy.sparse as sps
from sksparse.cholmod import cholesky

import importlib.util
spam_spec = importlib.util.find_spec("sksparse")
found = spam_spec is not None

from scipy.sparse.linalg import splu

if (found == True):
    print('kek')

def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
  
  n = A.shape[0]
  LU = splu(A,diag_pivot_thresh=0) # sparse LU decomposition
  
  if ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all(): # check the matrix A is positive definite.
    return LU.L.dot( sps.diags(LU.U.diagonal()**0.5) )
  else:
    sys.exit('The matrix is not positive definite')

def fastBlockInverse(Mh):
    
    if found == True:
        cholMh = cholesky(Mh)
        N = cholMh.L()
        Pv = cholMh.P()
        P = sps.csc_matrix((np.ones(Pv.size),(np.r_[0:Pv.size],Pv)), shape = (Pv.size,Pv.size))
    else:
        spluMh = splu(Mh)
        N = spluMh.L
        CC = spluMh.perm_c        
        Pv = spluMh.U
        P = sps.csc_matrix((np.ones(Pv.size),(np.r_[0:Pv.size],Pv)), shape = (Pv.size,Pv.size))
        
    # Extracting diagonals seems to be fastest with csc
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
    
    block_ends = np.r_[0,block_ends+1]
    
    elapsed = time.time()-tm; print('Preparing lists {:4.8f} seconds.'.format(elapsed))
    #####################################################################################
    
    
    #####################################################################################
    # Inversion of the blocks (naive version, keeping for sanity checks, who knows amirite?)
    #####################################################################################
    # iN = sps.lil_matrix(N.shape)
    
    # tm = time.time()
    # for i,ii in enumerate(block_ends[:-1]):
        
    #     C = N[block_ends[i]:block_ends[i+1],
    #           block_ends[i]:block_ends[i+1]].toarray()
        
    #     iC = np.linalg.inv(C)
        
    #     iN[block_ends[i]:block_ends[i+1],
    #         block_ends[i]:block_ends[i+1]] = iC
        
    # elapsed = time.time()-tm; print('Inverting naively took {:4.8f} seconds.'.format(elapsed))
    #####################################################################################    
    
    
    #####################################################################################
    # Inversion of the blocks, 2nd try.
    #####################################################################################
    
    tm = time.time()
    data_iN,indices_iN,indptr_iN = createIndicesInversion(N.data,N.indices,N.indptr,block_ends)
    iN = sps.csc_matrix((data_iN, indices_iN, indptr_iN), shape = N.shape)
    elapsed = time.time()-tm; print('Took {:4.8f} seconds.'.format(elapsed))
    
    iMh = P.T@(iN.T@iN)@P
    return iMh



@nb.njit()
def createIndicesInversion(dataN,indicesN,indptrN,block_ends):

    block_lengths = block_ends[1:]-block_ends[0:-1]
    
    sbl = np.sum(block_lengths)
    sbl2 = np.sum(block_lengths**2)
    
    C = np.zeros(sbl2)
    indices_iN = np.zeros(sbl2, dtype = np.int64)
    indptr_iN = np.zeros(sbl+1, dtype = np.int64)
    
    bli = 0; blis = 0; blis2 = 0
    
    for i in nb.prange(len(block_lengths)):
        
        blis = blis + bli
        blis2 = blis2 + bli**2
        
        bli = block_lengths[i]
        bei = block_ends[i]
        
        blis2p1 = blis2 + bli**2
        
        CC = np.zeros(shape = (bli,bli), dtype = np.float64)
        
        for k in range(bli):
            in_k = np.arange(start = indptrN[bei+k], stop = indptrN[bei+k+1], step = 1, dtype = np.int64)
            for j,jj in enumerate(in_k):
                CC[k,indicesN[jj]-bei] = dataN[jj]
                
            indptr_iN[k+blis+1] = blis2+bli*(k+1)
            indices_iN[blis2+bli*np.repeat(k,bli)+np.arange(0,bli)] = np.arange(bei,bei+bli)
        
        # This is needed (instead of above) if u wanna use parallel=True. Note: thats still slow af.
        # indptr_iN[blis+1+np.arange(0,k+1)] = blis2+bli*(np.arange(0,k+1)+1)
        
        iCCflat = np.linalg.inv(CC).flatten()
        C[blis2:blis2p1] = iCCflat
        
    return C,indices_iN,indptr_iN