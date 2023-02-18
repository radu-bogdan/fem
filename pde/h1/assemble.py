
from scipy import sparse as sp
import numpy as npy
from .spaces import spaceInfo
from .. import basis
from .. import quadrature


def assemble(MESH,space,matrix,order=-1):
    
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH,space)
    
    p = MESH.p;
    t = MESH.t; nt = t.shape[0]
    
    sizeM = MESH.FEMLISTS[space]['TRIG']['sizeM']

    phi = MESH.FEMLISTS[space]['TRIG']['phi']; lphi = len(phi)
    dphi = MESH.FEMLISTS[space]['TRIG']['dphi']; ldphi = len(dphi)
    
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
            qp =  MESH.FEMLISTS[space]['TRIG']['qp_we_M'][0]; 
            we =  MESH.FEMLISTS[space]['TRIG']['qp_we_M'][1]; nqp = len(we)
        
        ellmatsB = npy.zeros((nqp*nt,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
        
        for j in range(lphi):
            for i in range(nqp):
                ellmatsB[i*nt:(i+1)*nt,j] = phi[j](qp[0,i],qp[1,i])
        
        B = sparse(im,jm,ellmatsB,sizeM,nqp*nt)
        return B

    #####################################################################################
    # Stiffness matrices
    #####################################################################################
    
    if matrix == 'K':
        if order == -1:
            qp =  MESH.FEMLISTS[space]['TRIG']['qp_we_K'][0]; 
            we =  MESH.FEMLISTS[space]['TRIG']['qp_we_K'][1]; nqp = len(we)
        
        ellmatsBKx = npy.zeros((nqp*nt,ldphi))
        ellmatsBKy = npy.zeros((nqp*nt,ldphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(ldphi,1)).T
        
        for j in range(ldphi):
            for i in range(nqp):
                dphii = dphi[j](qp[0,i],qp[1,i])
                ellmatsBKx[i*nt:(i+1)*nt,j] = 1/detA*(A11*dphii[0]-A10*dphii[1])
                ellmatsBKy[i*nt:(i+1)*nt,j] = 1/detA*(-A01*dphii[0]+A00*dphii[1])
        
        BKx = sparse(im,jm,ellmatsBKx,sizeM,nqp*nt)
        BKy = sparse(im,jm,ellmatsBKy,sizeM,nqp*nt)
        return BKx, BKy

# @nb.jit(cache=True)
def assembleB(MESH,space,matrix,shape,order=-1):
    
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH,space)
    
    p = MESH.p;
    e = MESH.e; ne = e.shape[0]
    
    phi =  MESH.FEMLISTS[space]['B']['phi']; lphi = len(phi)
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
            qp = MESH.FEMLISTS[space]['B']['qp_we_B'][0];
            we = MESH.FEMLISTS[space]['B']['qp_we_B'][1]; nqp = len(we)
        
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