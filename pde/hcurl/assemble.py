
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
    curlphi = MESH.FEMLISTS[space]['TRIG']['curlphi']; lcurlphi = len(curlphi)

    LIST_DOF = MESH.FEMLISTS[space]['TRIG']['LIST_DOF']
    DIRECTION_DOF = MESH.FEMLISTS[space]['TRIG']['DIRECTION_DOF']
    
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
    
    if matrix == 'M' or matrix == 'phi':
        if order == -1:
            qp = MESH.FEMLISTS[space]['TRIG']['qp_we_M'][0]; 
            we = MESH.FEMLISTS[space]['TRIG']['qp_we_M'][1]; nqp = len(we)
        
        ellmatsBx = npy.zeros((nqp*nt,lphi))
        ellmatsBy = npy.zeros((nqp*nt,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
        
        for j in range(lphi):
            for i in range(nqp):
                phii = phi[j](qp[0,i],qp[1,i])                
                ellmatsBx[i*nt:(i+1)*nt,j] = 1/detA*( A11*phii[0] -A10*phii[1])*DIRECTION_DOF[:,j]
                ellmatsBy[i*nt:(i+1)*nt,j] = 1/detA*(-A01*phii[0] +A00*phii[1])*DIRECTION_DOF[:,j]
        
        Bx = sparse(im,jm,ellmatsBx,sizeM,nqp*nt)
        By = sparse(im,jm,ellmatsBy,sizeM,nqp*nt)
        return Bx,By
    
    
    #####################################################################################
    # Stiffness matrix
    #####################################################################################
    
    if matrix == 'K' or matrix == 'curlphi':
        if order == -1:
            qp = MESH.FEMLISTS[space]['TRIG']['qp_we_K'][0]; 
            we = MESH.FEMLISTS[space]['TRIG']['qp_we_K'][1]; nqp = len(we)
        
        ellmatsK = npy.zeros((nqp*nt,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[0:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
        
        for j in range(lcurlphi):
            for i in range(nqp):
                ellmatsK[i*nt:(i+1)*nt,j] = 1/detA*curlphi[j](qp[0,i],qp[1,i])*DIRECTION_DOF[:,j]
        
        K = sparse(im,jm,ellmatsK,sizeM,nqp*nt)
        return K
    

def assembleB(MESH,space,matrix,shape,order = -1, edges = npy.empty(0)):
    
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH,space)
    
    if not hasattr(MESH, 'Boundary_EdgeOrientation'):
        MESH.makeBEO()
    
    p = MESH.p
    e = MESH.e; ne = e.shape[0]
    
    phi = MESH.FEMLISTS[space]['B']['phi']; lphi = len(phi)
    
    if edges.size == 0:
        e = MESH.e
    else:
        e = MESH.EdgesToVertices[edges,:]
        
    ne = e.shape[0]
    LIST_DOF = MESH.FEMLISTS[space]['B']['LIST_DOF']
    
    if order != -1:
        qp,we = quadrature.one_d(order); nqp = len(we)
            
    #####################################################################################
    # Mappings
    #####################################################################################
        
    e0 = e[:,0]; e1 = e[:,1]
    A0 = p[e1,0]-p[e0,0]; A1 = p[e1,1]-p[e0,1]
    detA = npy.sqrt(A0**2+A1**2)
    
    if edges.size == 0:
        detA = detA*MESH.Boundary_EdgeOrientation
    
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
                ellmatsB[i*ne:(i+1)*ne,j] = 1/npy.abs(detA)*phi[j](qp[i])
        
        B = sparse(im,jm,ellmatsB,shape,nqp*ne)
        return B

def assembleE(MESH,space,matrix,order=-1):
    
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH,space)
    
    p = MESH.p; nt = MESH.nt
    e = MESH.EdgesToVertices; ne = e.shape[0]
    
    phi =  MESH.FEMLISTS[space]['B']['phi']; lphi = len(phi)
    LIST_DOF = MESH.FEMLISTS[space]['B']['LIST_DOF_E']
    phi_T =  MESH.FEMLISTS[space]['TRIG']['phi']; lphi_T = len(phi_T)
    
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
        
        ellmatsB = npy.zeros((nqp*ne,lphi_T))
        
        
        x = LIST_DOF
        # x = MESH.IntEdgesToTriangles[:,0]
        y = npy.c_[0:ne*nqp]
        
        im = npy.tile(x,(nqp,1))
        jm = npy.tile(y.reshape(ne,nqp).T.flatten(),(lphi,1)).T
        
        for j in range(lphi):
            for i in range(nqp):
                ellmatsB[i*ne:(i+1)*ne,j] = 1/npy.abs(detA)*phi[j](qp[i])
        
        B = sparse(im,jm,ellmatsB,3*nt,nqp*ne)
        # B = sparse(im,jm,ellmatsB,lphi*nt,nqp*ne)
        return B
        
    
def sparse(i, j, v, m, n):
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape = (m, n))