
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
    LIST_DOF = MESH.FEMLISTS[space]['TRIG']['LIST_DOF']
    
    if order != -1:
        qp,we =  quadrature.dunavant(order); nqp = len(we)

    #####################################################################################
    # Mappings
    #####################################################################################

    # t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]
    # A00 = p[t1,0]-p[t0,0]; A01 = p[t2,0]-p[t0,0]
    # A10 = p[t1,1]-p[t0,1]; A11 = p[t2,1]-p[t0,1]
    # detA = A00*A11-A01*A10
    
    #####################################################################################
    # Mass matrix
    #####################################################################################
    
    if matrix == 'M':
        if order == -1:
            qp = MESH.FEMLISTS[space]['TRIG']['qp_we_M'][0]; 
            we = MESH.FEMLISTS[space]['TRIG']['qp_we_M'][1]; nqp = len(we)
        
        ellmatsB = npy.zeros((nqp*nt,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
        
        for j in range(lphi):
            for i in range(nqp):
                ellmatsB[i*nt:(i+1)*nt,j] = phi[j](qp[0,i],qp[1,i])
        
        B = sparse(im,jm,ellmatsB,sizeM,nqp*nt)
        return B

def assemble1d(MESH,space,matrix,order=-1):
        
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH,space)
    
    p = MESH.p; nt = MESH.nt
    e = MESH.EdgesToVertices[MESH.NonSingle_Edges,:]; ne = e.shape[0]
    
    # sizeM = MESH.FEMLISTS[space]['TRIG']['sizeM']
    # phi = MESH.FEMLISTS[space]['TRIG']['phi']; lphi = len(phi)
    # LIST_DOF = MESH.FEMLISTS[space]['TRIG']['LIST_DOF']
    
    phi =  MESH.FEMLISTS[space]['B']['phi']; lphi = len(phi)
    LIST_DOF = MESH.FEMLISTS[space]['B']['LIST_DOF']
    sizeM = MESH.FEMLISTS[space]['TRIG']['sizeM']
    
    if order != -1:
        qp,we =  quadrature.dunavant(order); nqp = len(we)

    #####################################################################################
    # Mappings
    #####################################################################################

    # t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]
    # A00 = p[t1,0]-p[t0,0]; A01 = p[t2,0]-p[t0,0]
    # A10 = p[t1,1]-p[t0,1]; A11 = p[t2,1]-p[t0,1]
    # detA = A00*A11-A01*A10
    
    #####################################################################################
    # Mass matrix
    #####################################################################################
    
    if matrix == 'M':
        if order == -1:
            qp = MESH.FEMLISTS[space]['TRIG']['qp_we_M'][0]; 
            we = MESH.FEMLISTS[space]['TRIG']['qp_we_M'][1]; nqp = len(we)
        
        ellmatsB = npy.zeros((nqp*nt,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
        
        for j in range(lphi):
            for i in range(nqp):
                ellmatsB[i*nt:(i+1)*nt,j] = phi[j](qp[0,i],qp[1,i])
        
        B = sparse(im,jm,ellmatsB,sizeM,nqp*nt)
        return B

def sparse(i, j, v, m, n):
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape = (m, n))