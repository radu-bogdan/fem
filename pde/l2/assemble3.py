
from scipy import sparse as sp
import numpy as npy
from .spaces import spaceInfo
from .. import basis
from .. import quadrature

def assemble3(MESH,space,matrix,order=-1):
        
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH,space)
    
    p = MESH.p;
    t = MESH.t; nt = t.shape[0]
    
    sizeM = MESH.FEMLISTS[space]['TET']['sizeM']
    phi = MESH.FEMLISTS[space]['TET']['phi']; lphi = len(phi)
    dphi = MESH.FEMLISTS[space]['TET']['dphi']; ldphi = len(dphi)
    
    LIST_DOF = MESH.FEMLISTS[space]['TET']['LIST_DOF']
    
    if order != -1:
        qp,we = quadrature.keast(order); nqp = len(we)
    
    #####################################################################################
    # Mass matrix
    #####################################################################################
    
    if matrix == 'M':
        if order == -1:
            qp = MESH.FEMLISTS[space]['TET']['qp_we_M'][0]; 
            we = MESH.FEMLISTS[space]['TET']['qp_we_M'][1]; nqp = len(we)
        
        ellmatsB = npy.zeros((nqp*nt,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
        
        for j in range(lphi):
            for i in range(nqp):
                ellmatsB[i*nt:(i+1)*nt,j] = phi[j](qp[0,i],qp[1,i],qp[2,i])
        
        B = sparse(im,jm,ellmatsB,sizeM,nqp*nt)
        return B
    
    #####################################################################################
    # Stiffness matrices
    #####################################################################################
    
    if matrix == 'K':
        if order == -1:
            qp =  MESH.FEMLISTS[space]['TET']['qp_we_K'][0]; 
            we =  MESH.FEMLISTS[space]['TET']['qp_we_K'][1]; nqp = len(we)
        
        ellmatsBKx = npy.zeros((nqp*nt,ldphi))
        ellmatsBKy = npy.zeros((nqp*nt,ldphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[:nt*nqp].reshape(nt,nqp).T.flatten(),(ldphi,1)).T
        
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
    
def assembleE3(MESH,space,matrix,order=0):
        
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH,space)
    
    p = MESH.p; nt = MESH.nt
    e = MESH.EdgesToVertices; ne = e.shape[0]
    
    phi =  MESH.FEMLISTS[space]['B']['phi']; lphi = len(phi)
    LIST_DOF = MESH.FEMLISTS[space]['B']['LIST_DOF']
    sizeM = MESH.FEMLISTS[space]['B']['sizeM']
    
    qp,we = quadrature.one_d(order); nqp = len(we)

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
    
    LIST_DOF_L2E = npy.r_[0:ne]
    
    if matrix == 'M':
        
        ellmatsB = npy.zeros((nqp*ne,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        # jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
        jm = npy.tile(npy.tile(LIST_DOF_L2E*nqp,nqp) + npy.arange(nqp).repeat(ne),(lphi,1)).T
        
        for j in range(lphi):
            for i in range(nqp):
                ellmatsB[i*ne:(i+1)*ne,j] = phi[j](qp[i])
                # ellmatsB[i*ne:(i+1)*ne,j] = phi[j](qp[i])*MESH.EdgeDirectionTrig ???
        
        B = sparse(im,jm,ellmatsB,sizeM,nqp*ne)
        return B

def sparse(i, j, v, m, n):
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape = (m, n))