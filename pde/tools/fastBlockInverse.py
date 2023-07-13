import sys
import time
import numpy as np
import numba as nb
import scipy.sparse as sps
from numba import float64,float32,int64

import importlib.util
spam_spec = importlib.util.find_spec("sksparse")
found = spam_spec is not None

from scipy.sparse.linalg import splu

if found == False:
    # @profile()
    def fastBlockInverse(Mh):
        spluMh = splu(Mh)
        L = spluMh.L; U = spluMh.U

        Pv2 = spluMh.perm_r
        Pv3 = spluMh.perm_c

        P2 = sps.csc_matrix((np.ones(Pv2.size),(np.r_[0:Pv2.size],Pv2)), shape = (Pv2.size,Pv2.size))
        P3 = sps.csc_matrix((np.ones(Pv3.size),(np.r_[0:Pv3.size],Pv3)), shape = (Pv3.size,Pv3.size))
        
        L = L.tocsc()
        UT = (U.T).tocsc()
        
        
        #####################################################################################
        # Find indices where the blocks begin/end
        #####################################################################################
        tm = time.time()
        
        L_diag = L.diagonal(k=-1) # Nebendiagonale anfangen
        block_ends_L = np.r_[np.argwhere(abs(L_diag)==0)[:,0],L.shape[0]-1]
        
        for i in range(L.shape[0]):
            L_diag = np.r_[L.diagonal(k=-(i+2)),np.zeros(i+2)]
            
            for j in range(i+1):
                arg = np.argwhere(abs(L_diag[block_ends_L-j])>0)[:,0]
                block_ends_L = np.delete(block_ends_L,arg).copy()
                
            if np.linalg.norm(L_diag)==0: break
        
        block_ends_L = np.r_[0,block_ends_L+1]
        
        #####################################################################################
        
        
        #####################################################################################
        # Find indices where the blocks begin/end
        #####################################################################################
        
        UT_diag = UT.diagonal(k=-1) # Nebendiagonale anfangen
        block_ends_UT = np.r_[np.argwhere(abs(UT_diag)==0)[:,0],UT.shape[0]-1]
        
        for i in range(UT.shape[0]):
            UT_diag = np.r_[UT.diagonal(k=-(i+2)),np.zeros(i+2)]
            
            for j in range(i+1):
                arg = np.argwhere(abs(UT_diag[block_ends_UT-j])>0)[:,0]
                block_ends_UT = np.delete(block_ends_UT,arg).copy()
                
            if np.linalg.norm(UT_diag)==0: break
        
        block_ends_UT = np.r_[0,block_ends_UT+1]
        
        # elapsed = time.time()-tm; print('Preparing lists {:4.8f} seconds.'.format(elapsed))
        #####################################################################################
        
        tm = time.time()
        data_iUT,indices_iUT,indptr_iUT = createIndicesInversion(UT.data,UT.indices,UT.indptr,block_ends_UT)
        iUT = sps.csc_matrix((data_iUT, indices_iUT, indptr_iUT), shape = UT.shape)
        
        data_iL,indices_iL,indptr_iL = createIndicesInversion(L.data,L.indices,L.indptr,block_ends_L)
        iL = sps.csc_matrix((data_iL, indices_iL, indptr_iL), shape = L.shape)
        # elapsed = time.time()-tm; print('Took {:4.8f} seconds.'.format(elapsed))
        
        iMh = P3@(iUT.T@iL)@P2.T
        return iMh#P3@(iUT.T@iL)@P2.T

if found == True:
    from sksparse.cholmod import cholesky
    def fastBlockInverse(Mh):
        
        cholMh = cholesky(Mh)
        N = cholMh.L()
        Pv = cholMh.P()
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



@nb.njit(cache = True, parallel = True, fastmath = False)
def createIndicesInversion(dataN,indicesN,indptrN,block_ends) -> (float64[:],int64[:],int64[:]):

    block_lengths = block_ends[1:]-block_ends[0:-1]
    
    sbl = np.sum(block_lengths)+1
    sbl2 = np.sum(block_lengths**2)
    
    blicum = np.zeros(block_lengths.size+1, dtype = np.int64)
    bli2cum = np.zeros(block_lengths.size+1, dtype = np.int64)
    
    for z in range(block_lengths.size):
        blicum[z+1] = blicum[z] + block_lengths[z]
        bli2cum[z+1] = bli2cum[z] + block_lengths[z]**2
    
    # blicum = np.cumsum(block_lengths)
    C = np.zeros(sbl2)
    indptr_iN = np.zeros(sbl, dtype = int64)
    indices_iN = np.zeros(sbl2, dtype = int64)
    
    blis = 0; blis2 = 0
    
    for i in nb.prange(block_lengths.size):
        
        # if i==0: bli = 0
        # else: bli = block_lengths[i-1]
        
        blis = blicum[i]
        blis2 = bli2cum[i]
        
        #blis = blis + bli
        #blis2 = blis2 + bli**2
        
        bli = block_lengths[i]
        bei = block_ends[i]
        
        blis2p1 = blis2 + bli**2
        
        CC = np.zeros(shape = (bli,bli), dtype = np.float64)
        
        # print(blis+1+np.arange(0,bli),blis2+bli*(np.arange(0,bli)+1))
        # indptr_iN[blis+1+np.arange(0,bli+1)] = blis2+bli*(np.arange(0,bli+1)+1)
        # indptr_iN2[blis+1+np.arange(0,bli)] = blis2+bli*(np.arange(0,bli)+1)
        
        for k in range(bli):
            # # in_k = np.arange(start = indptrN[bei+k], stop = indptrN[bei+k+1], step = 1, dtype = np.int32)
            in_k = np.arange(indptrN[bei+k],indptrN[bei+k+1])
            for _,jj in enumerate(in_k):
                CC[k,indicesN[jj]-bei] = dataN[jj]
                                
            indptr_iN[k+blis+1] = blis2+bli*(k+1)
            indices_iN[blis2+bli*np.repeat(k,bli)+np.arange(0,bli)] = np.arange(bei,bei+bli)
            
            # print(blis2+bli*np.repeat(k,bli)+np.arange(0,bli))
        
        # This is needed (instead of above) if u wanna use parallel=True. Note: thats still slow af.
        # indptr_iN[blis+1+np.arange(0,k+1)] = blis2+bli*(np.arange(0,k+1)+1)
        
        iCCflat = np.linalg.inv(CC).flatten()
        # iCCflat = CC.flatten()
        C[blis2:blis2p1] = iCCflat
    
    # indptr_iN = indptr_iN2
    # print(indptr_iN2)
    # print(np.max(np.abs(indptr_iN2-indptr_iN)),indptr_iN[0])
    return C,indices_iN,indptr_iN

# createIndicesInversion.parallel_diagnostics(level=4)

# Md = np.load('lol.npz',allow_pickle=True)['Md'].tolist()

# if __name__ == '__main__':
#     tm = time.monotonic()
#     for i in range(1):
#         iMd = fastBlockInverse(Md)
#     print('Inverting took ', time.monotonic()-tm)
        
#     print(sps.linalg.norm(Md@iMd-sps.eye(Md.shape[0]),np.inf))