
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
                detA = MESH.detA(qp[0,i],qp[1,i])
                iJF00 = MESH.iJF00(qp[0,i],qp[1,i]); iJF01 = MESH.iJF01(qp[0,i],qp[1,i]);
                iJF10 = MESH.iJF10(qp[0,i],qp[1,i]); iJF11 = MESH.iJF11(qp[0,i],qp[1,i]);
                
                ellmatsBKx[i*nt:(i+1)*nt,j] = iJF00*dphii[0]+iJF10*dphii[1]
                ellmatsBKy[i*nt:(i+1)*nt,j] = iJF01*dphii[0]+iJF11*dphii[1]
        
        BKx = sparse(im,jm,ellmatsBKx,sizeM,nqp*nt)
        BKy = sparse(im,jm,ellmatsBKy,sizeM,nqp*nt)
        return BKx, BKy
        
    if matrix == 'div':
        return 0

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
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape=(m, n))