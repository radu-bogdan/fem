
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
    if space == 'P1': 
        INFO['TRIG']['space'] = 'P1'
        INFO['TRIG']['space_dx'] = 'P0'
        INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
        INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
        INFO['sizeM'] = MESH.np
        INFO['sizeD'] = MESH.nt
        INFO['qp_we_B'] = quadrature.one_d(order = 2)
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P2': 
        INFO['TRIG']['space'] = 'P2'
        INFO['TRIG']['space_dx'] = 'P1'
        INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 4)
        # INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
        INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 2)
        INFO['sizeM'] = MESH.np + MESH.NoEdges
        INFO['sizeD'] = 3*MESH.nt
        INFO['qp_we_B'] = quadrature.one_d(order = 5) # 4 would suffice
    ###########################################################################
    
    return INFO

def assembleD(MESH, order, coeff = lambda x,y : 1+0*x*y, regions = npy.empty(0)):
    
    if regions.size == 0:
        regions = MESH.RegionsT
    
    indices = npy.argwhere(npy.in1d(MESH.RegionsT,regions))[:,0]

    p = MESH.p;
    t = MESH.t[indices,:]; nt = t.shape[0]
    
    #####################################################################################
    # Mappings
    #####################################################################################

    t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]
    A00 = p[t1,0]-p[t0,0]; A01 = p[t2,0]-p[t0,0]
    A10 = p[t1,1]-p[t0,1]; A11 = p[t2,1]-p[t0,1]
    
    #####################################################################################
    # Custom config matrix
    #####################################################################################
    
    qp,we = quadrature.dunavant(order); nqp = len(we)
    ellmatsD = npy.zeros((nqp*nt))
    
    iD = npy.r_[0:nqp*MESH.nt].reshape(MESH.nt,nqp).T
    iD = iD[:,indices]
        
    for i in range(nqp):
        qpT_i_1 = A00*qp[0,i]+A01*qp[1,i]+p[t0,0]
        qpT_i_2 = A10*qp[0,i]+A11*qp[1,i]+p[t0,1]
        ellmatsD[i*nt:(i+1)*nt] = coeff(qpT_i_1,qpT_i_2)
    
    D = sparse(iD,iD,ellmatsD,nqp*MESH.nt,nqp*MESH.nt)
    return D



def assembleDB(MESH, order, coeff = lambda x,y : 1+0*x*y, edges = npy.empty(0)):
    
    if edges.size == 0:
        edges = MESH.Boundary.Region
    
    indices = npy.argwhere(npy.in1d(MESH.Boundary.Region,edges))[:,0]

    p = MESH.p;    
    e = MESH.e[indices,:]; ne = e.shape[0]
    
    #####################################################################################
    # Mappings
    #####################################################################################

    e0 = e[:,0]; e1 = e[:,1]
    A0 =  p[e1,0]-p[e0,0]; A1 =  p[e1,1]-p[e0,1]
    
    #####################################################################################
    # Custom config matrix
    #####################################################################################
    
    qp,we = quadrature.one_d(order); nqp = len(we)
    ellmatsD = npy.zeros((nqp*ne))
    
    iD = npy.r_[0:nqp*MESH.ne].reshape(MESH.ne,nqp).T
    iD = iD[:,indices]
        
    for i in range(nqp):
        qpT_i_1 = A0*qp[i] + p[e0,0]
        qpT_i_2 = A1*qp[i] + p[e0,1]
        ellmatsD[i*ne:(i+1)*ne] = coeff(qpT_i_1,qpT_i_2)
    
    D = sparse(iD,iD,ellmatsD,nqp*MESH.ne,nqp*MESH.ne)
    return D



def assemble(MESH,space,matrix,order=-1):
    
    INFO = get_info(MESH,space)
    BASIS = basis()
    
    p = MESH.p;
    t = MESH.t; nt = t.shape[0]
    
    sizeM = INFO['sizeM']
    spaceTrig = INFO['TRIG']['space'];

    phi = BASIS[spaceTrig]['TRIG']['phi']; lphi = len(phi)
    dphi = BASIS[spaceTrig]['TRIG']['dphi']; ldphi = len(dphi)
    LIST_DOF = MESH.FEMLISTS[spaceTrig]['TRIG']['LIST_DOF']
    
    if order != -1:
        qp,we = quadrature.dunavant(order); nqp = len(we)

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
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
        iD = npy.r_[0:nqp*nt].reshape(nt,nqp).T
        
        for j in range(lphi):
            for i in range(nqp):
                ellmatsB[i*nt:(i+1)*nt,j] = phi[j](qp[0,i],qp[1,i])
                ellmatsD[i*nt:(i+1)*nt] = 1/2*npy.abs(detA)*we[i]
        
        B = sparse(im,jm,ellmatsB,sizeM,nqp*nt)
        D = sparse(iD,iD,ellmatsD,nqp*nt,nqp*nt)
        return B, D

    #####################################################################################
    # Stiffness matrices
    #####################################################################################
    
    if matrix == 'K':
        if order == -1:
            qp = INFO['TRIG']['qp_we_K'][0]; we = INFO['TRIG']['qp_we_K'][1]; nqp = len(we)
        
        ellmatsBKx = npy.zeros((nqp*nt,ldphi))
        ellmatsBKy = npy.zeros((nqp*nt,ldphi))
        ellmatsD = npy.zeros((nqp*nt))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(ldphi,1)).T
        iD = npy.r_[0:nqp*nt].reshape(nt,nqp).T
        
        for j in range(ldphi):
            for i in range(nqp):
                dphii = dphi[j](qp[0,i],qp[1,i])
                ellmatsBKx[i*nt:(i+1)*nt,j] = 1/detA*(A11*dphii[0]-A10*dphii[1])
                ellmatsBKy[i*nt:(i+1)*nt,j] = 1/detA*(-A01*dphii[0]+A00*dphii[1])
                ellmatsD[i*nt:(i+1)*nt] = 1/2*npy.abs(detA)*we[i]
        
        BKx = sparse(im,jm,ellmatsBKx,sizeM,nqp*nt)
        BKy = sparse(im,jm,ellmatsBKy,sizeM,nqp*nt)
        DK = sparse(iD,iD,ellmatsD,nqp*nt,nqp*nt)
        return BKx, BKy, DK



def assembleB(MESH,space,matrix,shape,order=-1):
    
    INFO = get_info(MESH,space)
    BASIS = basis()
    
    p = MESH.p;
    e = MESH.e; ne = e.shape[0]
    
    phi = BASIS[space]['B']['phi']; lphi = len(phi)
    
    LIST_DOF = MESH.FEMLISTS[space]['B']['LIST_DOF']
    
    if order != -1:
        qp,we = quadrature.one_d(order); nqp = len(we)
            
    #####################################################################################
    # Mappings
    #####################################################################################
        
    e0 = e[:,0]; e1 = e[:,1]
    A0 =  p[e1,0]-p[e0,0]; A1 =  p[e1,1]-p[e0,1]
    detA = npy.sqrt(A0**2+A1**2)
    
    #####################################################################################
    # Mass matrix (over the edge)
    #####################################################################################

    if matrix == 'M':
        if order == -1:
            qp = INFO['qp_we_B'][0]; we = INFO['qp_we_B'][1]; nqp = len(we)
        
        ellmatsB = npy.zeros((nqp*ne,lphi))
        ellmatsD = npy.zeros((nqp*ne))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:ne*nqp].reshape(ne,nqp).T.flatten(),(lphi,1)).T
        iD = npy.r_[0:nqp*ne].reshape(ne,nqp).T
        
        for j in range(lphi):
            for i in range(nqp):
                ellmatsB[i*ne:(i+1)*ne,j] = phi[j](qp[i])
                ellmatsD[i*ne:(i+1)*ne] = npy.abs(detA)*we[i]
        
        B = sparse(im,jm,ellmatsB,shape[0],nqp*ne)
        D = sparse(iD,iD,ellmatsD,nqp*ne,nqp*ne)
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
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape=(m, n))
    # return sp.csr_matrix((v.flatten(), (i.flatten(), j.flatten())), shape=(m, n))