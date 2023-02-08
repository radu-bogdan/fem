
from scipy import sparse as sp
import numpy as npy
from .spaces import spaceInfo
from .. import basis
from .. import quadrature


def assemble(MESH,space,matrix,order=-1):
    
    INFO = spaceInfo(MESH,space)
    # BASIS = basis()
    
    p = MESH.p;
    t = MESH.t; nt = t.shape[0]
    
    sizeM = INFO['sizeM']

    phi = INFO['TRIG']['phi']; lphi = len(phi)
    # dphi = INFO['TRIG']['dphi']; ldphi = len(dphi)
    LIST_DOF = MESH.FEMLISTS[space]['TRIG']['LIST_DOF']
    
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
        
        ellmatsB00 = npy.zeros((nqp*nt,lphi))
        ellmatsB01 = npy.zeros((nqp*nt,lphi))
        ellmatsB10 = npy.zeros((nqp*nt,lphi))
        ellmatsB11 = npy.zeros((nqp*nt,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
        
        for j in range(lphi):
            for i in range(nqp):
                phii = phi[j](qp[0,i],qp[1,i])
                ellmatsB00[i*nt:(i+1)*nt,j] = 1/(detA*detA)*(A00*(A00*phii[0][0] + A01*phii[0][1]) + A01*(A00*phii[1][0] + A01*phii[1][1]))
                ellmatsB01[i*nt:(i+1)*nt,j] = 1/(detA*detA)*(A00*(A10*phii[0][0] + A11*phii[0][1]) + A01*(A10*phii[1][0] + A11*phii[1][1]))
                ellmatsB10[i*nt:(i+1)*nt,j] = 1/(detA*detA)*(A10*(A00*phii[0][0] + A01*phii[0][1]) + A11*(A00*phii[1][0] + A01*phii[1][1]))
                ellmatsB11[i*nt:(i+1)*nt,j] = 1/(detA*detA)*(A10*(A10*phii[0][0] + A11*phii[0][1]) + A11*(A10*phii[1][0] + A11*phii[1][1]))
        
        B00 = sparse(im,jm,ellmatsB00,sizeM,nqp*nt)
        B01 = sparse(im,jm,ellmatsB01,sizeM,nqp*nt)
        B10 = sparse(im,jm,ellmatsB10,sizeM,nqp*nt)
        B11 = sparse(im,jm,ellmatsB11,sizeM,nqp*nt)
        return B00,B01,B10,B11

    # #####################################################################################
    # # Stiffness matrices
    # #####################################################################################
    
    # if matrix == 'K':
    #     if order == -1:
    #         qp = INFO['TRIG']['qp_we_K'][0]; we = INFO['TRIG']['qp_we_K'][1]; nqp = len(we)
        
    #     ellmatsBKx = npy.zeros((nqp*nt,ldphi))
    #     ellmatsBKy = npy.zeros((nqp*nt,ldphi))
        
    #     im = npy.tile(LIST_DOF,(nqp,1))
    #     jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(ldphi,1)).T
        
    #     for j in range(ldphi):
    #         for i in range(nqp):
    #             dphii = dphi[j](qp[0,i],qp[1,i])
    #             ellmatsBKx[i*nt:(i+1)*nt,j] = 1/detA*(A11*dphii[0]-A10*dphii[1])
    #             ellmatsBKy[i*nt:(i+1)*nt,j] = 1/detA*(-A01*dphii[0]+A00*dphii[1])
        
    #     BKx = sparse(im,jm,ellmatsBKx,sizeM,nqp*nt)
    #     BKy = sparse(im,jm,ellmatsBKy,sizeM,nqp*nt)
    #     return BKx, BKy

# @nb.jit(cache=True)
def assembleB(MESH,space,matrix,shape,order=-1):
    
    INFO = spaceInfo(MESH,space)
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
    A0 = p[e1,0]-p[e0,0]; A1 = p[e1,1]-p[e0,1]
    detA = npy.sqrt(A0**2+A1**2)
    
    #####################################################################################
    # Mass matrix (over the edge)
    #####################################################################################

    if matrix == 'M':
        if order == -1:
            qp = INFO['qp_we_B'][0]; we = INFO['qp_we_B'][1]; nqp = len(we)
        
        ellmatsB = npy.zeros((nqp*ne,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:ne*nqp].reshape(ne,nqp).T.flatten(),(lphi,1)).T
        
        for j in range(lphi):
            for i in range(nqp):
                ellmatsB[i*ne:(i+1)*ne,j] = phi[j](qp[i])
        
        B = sparse(im,jm,ellmatsB,shape[0],nqp*ne)
        return B

def sparse(i, j, v, m, n):
    # return sp.csc_matrix((v.flatten(order='F'), (i.flatten(order='F'), j.flatten(order='F'))), shape=(m, n))
    # return sp.csr_matrix((v.flatten(), (i.flatten(), j.flatten())), shape=(m, n))
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape=(m, n))