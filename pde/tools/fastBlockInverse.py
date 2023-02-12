import time
import numpy as np
import numba as nb
import scipy.sparse as sps

def fastBlockInverse(N):
    
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
    return iN



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