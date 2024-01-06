
from scipy import sparse as sp
import numpy as npy
from .spaces import spaceInfo
from .. import basis
from .. import quadrature
from ..tools import getIndices


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
    
def assembleE(MESH,space,matrix,order=1):
    
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH,space)
    
    p = MESH.p; 
    t = MESH.t; nt = MESH.nt
    
    phi =  MESH.FEMLISTS[space]['TRIG']['phi']; lphi = len(phi)
    sizeM = MESH.FEMLISTS[space]['TRIG']['sizeM']
        
    LIST_DOF = MESH.FEMLISTS[space]['TRIG']['LIST_DOF']
    # DIRECTION_DOF = MESH.FEMLISTS[space]['TRIG']['DIRECTION_DOF']
    
    qp,we = quadrature.one_d(order); nqp = len(we)
    
    # on edge2: (0,0) -> (1,0)
    qp2 = npy.c_[1-qp,0*qp].T
    # on edge1: (0,0) -> (0,1)
    qp1 = npy.c_[0*qp,qp].T
    # on edge0: (0,1) -> (1,0)
    qp0 = npy.c_[qp,1-qp].T
    
    #####################################################################################
    # Mappings
    #####################################################################################
    
    t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]
    A00 = p[t1,0]-p[t0,0]; A01 = p[t2,0]-p[t0,0];
    A10 = p[t1,1]-p[t0,1]; A11 = p[t2,1]-p[t0,1];
    detA = A00*A11-A01*A10
    
    #####################################################################################
    # Mass matrix (over the edge)
    #####################################################################################
    
    if matrix == 'M':
        
        ellmatsB0 = npy.zeros((nqp*nt,lphi))
        ellmatsB1 = npy.zeros((nqp*nt,lphi))
        ellmatsB2 = npy.zeros((nqp*nt,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        
        ind0 = npy.argwhere(MESH.EdgeDirectionTrig[:,0]==-1)
        ind1 = npy.argwhere(MESH.EdgeDirectionTrig[:,1]==-1)
        ind2 = npy.argwhere(MESH.EdgeDirectionTrig[:,2]==-1)
        
        indices = npy.tile(npy.arange(nqp),(nt,1))
        
        indices0 = indices.copy(); indices1 = indices.copy(); indices2 = indices.copy()
        
        indices0[ind0,:] = indices0[ind0,::-1]; 
        indices1[ind1,:] = indices1[ind1,::-1]; 
        indices2[ind2,:] = indices2[ind2,::-1];
        
        jm0 = npy.tile(npy.tile(MESH.TriangleToEdges[:,0]*nqp,nqp) + indices0.T.flatten(),(lphi,1)).T
        jm1 = npy.tile(npy.tile(MESH.TriangleToEdges[:,1]*nqp,nqp) + indices1.T.flatten(),(lphi,1)).T
        jm2 = npy.tile(npy.tile(MESH.TriangleToEdges[:,2]*nqp,nqp) + indices2.T.flatten(),(lphi,1)).T

        for j in range(lphi):
            for i in range(nqp):
                phii0 = phi[j](qp0[0,i],qp0[1,i])
                phii1 = phi[j](qp1[0,i],qp1[1,i])
                phii2 = phi[j](qp2[0,i],qp2[1,i])
                
                ellmatsB0[i*nt:(i+1)*nt,j] = phii0*MESH.EdgeDirectionTrig[:,0]
                ellmatsB1[i*nt:(i+1)*nt,j] = phii1*MESH.EdgeDirectionTrig[:,1]
                ellmatsB2[i*nt:(i+1)*nt,j] = phii2*MESH.EdgeDirectionTrig[:,2]
        
        ellmatsB0 = ellmatsB0*(npy.abs(ellmatsB0)>1e-12)
        ellmatsB1 = ellmatsB1*(npy.abs(ellmatsB1)>1e-12)
        ellmatsB2 = ellmatsB2*(npy.abs(ellmatsB2)>1e-12)
        
        B0 = sparse(im,jm0,ellmatsB0,sizeM,nqp*MESH.NoEdges)
        B1 = sparse(im,jm1,ellmatsB1,sizeM,nqp*MESH.NoEdges)
        B2 = sparse(im,jm2,ellmatsB2,sizeM,nqp*MESH.NoEdges)
        
        B0.eliminate_zeros()
        B1.eliminate_zeros()
        B2.eliminate_zeros()
        
        return B0,B1,B2

def assembleR(MESH, space, edges = '', listDOF = npy.empty(0)):
    
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH,space)
        
    if type(edges) == str:
        if edges == '':
            ind_edges = MESH.Boundary_Region
        else:
            ind_edges = getIndices(MESH.regions_1d,edges)
    else:
        if MESH.regions_1d == []:
            ind_edges = edges
        else:
            ind_edges = getIndices(MESH.regions_1d,edges)
            
    
    indices = npy.in1d(MESH.Boundary_Region,ind_edges)
    
    sizeM = MESH.FEMLISTS[space]['TRIG']['sizeM']
    LIST_DOF  = npy.unique(MESH.FEMLISTS[space]['B']['LIST_DOF'][indices,:])
    LIST_DOF2 = npy.setdiff1d(npy.arange(sizeM),LIST_DOF)
    
    if listDOF.size > 0:
        LIST_DOF = listDOF
    
    D = sp.eye(sizeM, format = 'csc')
    R1 = D[:,LIST_DOF]
    R2 = D[:,LIST_DOF2]
    
    return R1.T.tocsc(),R2.T.tocsc()


def sparse(i, j, v, m, n):
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape=(m, n))