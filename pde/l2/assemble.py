
from .. import quadrature
from .. import basis
# from .. import lists
import numpy as npy
from scipy import sparse as sp
# import matplotlib.pyplot as plt
# import time

def get_info(MESH,space):
    
    INFO = {}
    INFO['TRIG'] = {}
    
    ###########################################################################
    if space == 'P0': 
        INFO['TRIG']['space'] = 'P0'
        INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 0)
        INFO['sizeM'] = MESH.nt
        INFO['qp_we_B'] = quadrature.one_d(order = 2)
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P1d': 
        INFO['TRIG']['space'] = 'P1d'
        INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        INFO['sizeM'] = 3*MESH.nt
        INFO['qp_we_B'] = quadrature.one_d(order = 2)
    ###########################################################################
    
    return INFO

def assemble(MESH,space,matrix,order=-1):
    
    INFO = get_info(MESH,space)
    BASIS = basis()
    
    p = MESH.p;
    t = MESH.t; nt = t.shape[0]
    
    sizeM = INFO['sizeM']
    spaceTrig = INFO['TRIG']['space'];

    phi = BASIS[spaceTrig]['TRIG']['phi']; lphi = len(phi)
    H1_LIST_DOF = MESH.FEMLISTS[spaceTrig]['TRIG']['LIST_DOF']
    
    if order != -1:
        qp,we =  quadrature.dunavant(order); nqp = len(we)

    #####################################################################################
    # Mappings
    #####################################################################################

    t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]
    A00 = p[t1,0]-p[t0,0]; A01 = p[t2,0]-p[t0,0]
    A10 = p[t1,1]-p[t0,1]; A11 = p[t2,1]-p[t0,1]
    detA = A00*A11-A01*A10
    
    #####################################################################################
    # Mass matrix
    #####################################################################################
    
    if matrix == 'M':
        if order == -1:
            qp = INFO['TRIG']['qp_we_M'][0]; we = INFO['TRIG']['qp_we_M'][1]; nqp = len(we)
        
        ellmatsB = npy.zeros((nqp*nt,lphi))
        ellmatsD = npy.zeros((nqp*nt))
        
        im = npy.tile(H1_LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
        iD = npy.r_[0:nqp*nt].reshape(nt,nqp).T
        
        for j in range(lphi):
            for i in range(nqp):
                ellmatsB[i*nt:(i+1)*nt,j] = phi[j](qp[0,i],qp[1,i])
                ellmatsD[i*nt:(i+1)*nt] = 1/2*npy.abs(detA)*we[i]
        
        B = sparse(im,jm,ellmatsB,sizeM,nqp*nt)
        D = sparse(iD,iD,ellmatsD,nqp*nt,nqp*nt)
        return B, D

def sparse(i, j, v, m, n):
    """
    Create and compressing a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values 
            Size n1
        j: 1-D array representing the index 2 values 
            Size n1
        v: 1-D array representing the values 
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    # return sp.csc_matrix((v.flatten(order='F'), (i.flatten(order='F'), j.flatten(order='F'))), shape=(m, n))
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape = (m, n))
    # return sp.csr_matrix((v.flatten(), (i.flatten(), j.flatten())), shape=(m, n))