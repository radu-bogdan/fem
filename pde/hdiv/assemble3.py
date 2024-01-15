
from scipy import sparse as sp
import numpy as npy
from .spaces3 import spaceInfo
from ..tools import getIndices
from .. import basis
from .. import quadrature


def assemble3(MESH,space,matrix,order=-1):
    
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH,space)
    
    p = MESH.p;
    t = MESH.t; nt = t.shape[0]
    
    sizeM = MESH.FEMLISTS[space]['TET']['sizeM']

    phi = MESH.FEMLISTS[space]['TET']['phi']; lphi = len(phi)
    divphi = MESH.FEMLISTS[space]['TET']['divphi']; ldivphi = len(divphi)

    LIST_DOF = MESH.FEMLISTS[space]['TET']['LIST_DOF']
    DIRECTION_DOF = MESH.FEMLISTS[space]['TET']['DIRECTION_DOF']
    
    if order != -1:
        qp,we = quadrature.keast(order); nqp = len(we)
    
    #####################################################################################
    # Mass matrix
    #####################################################################################
    
    if matrix == 'M' or matrix == 'phi':
        if order == -1:
            qp = MESH.FEMLISTS[space]['TET']['qp_we_M'][0]; 
            we = MESH.FEMLISTS[space]['TET']['qp_we_M'][1]; nqp = len(we)
        
        ellmatsBx = npy.zeros((nqp*nt,lphi))
        ellmatsBy = npy.zeros((nqp*nt,lphi))
        ellmatsBz = npy.zeros((nqp*nt,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
        
        for j in range(lphi):
            for i in range(nqp):
                phii = phi[j](qp[0,i],qp[1,i],qp[2,i])
                detA = MESH.detA(qp[0,i],qp[1,i],qp[2,i])
                JF00 = MESH.JF00(qp[0,i],qp[1,i],qp[2,i]); JF01 = MESH.JF01(qp[0,i],qp[1,i],qp[2,i]); JF02 = MESH.JF02(qp[0,i],qp[1,i],qp[2,i]);
                JF10 = MESH.JF10(qp[0,i],qp[1,i],qp[2,i]); JF11 = MESH.JF11(qp[0,i],qp[1,i],qp[2,i]); JF12 = MESH.JF12(qp[0,i],qp[1,i],qp[2,i]);
                JF20 = MESH.JF20(qp[0,i],qp[1,i],qp[2,i]); JF21 = MESH.JF21(qp[0,i],qp[1,i],qp[2,i]); JF22 = MESH.JF22(qp[0,i],qp[1,i],qp[2,i]);
                
                ellmatsBx[i*nt:(i+1)*nt,j] = 1/(detA)*(JF00*phii[0]+JF01*phii[1]+JF02*phii[2])*DIRECTION_DOF[:,j]
                ellmatsBy[i*nt:(i+1)*nt,j] = 1/(detA)*(JF10*phii[0]+JF11*phii[1]+JF12*phii[2])*DIRECTION_DOF[:,j]
                ellmatsBz[i*nt:(i+1)*nt,j] = 1/(detA)*(JF20*phii[0]+JF21*phii[1]+JF22*phii[2])*DIRECTION_DOF[:,j]
        
        Bx = sparse(im,jm,ellmatsBx,sizeM,nqp*nt)
        By = sparse(im,jm,ellmatsBy,sizeM,nqp*nt)
        Bz = sparse(im,jm,ellmatsBz,sizeM,nqp*nt)
        return Bx,By,Bz
    
    
    #####################################################################################
    # Stiffness matrix
    #####################################################################################
    
    # todo
    if matrix == 'K' or matrix == 'divphi':
        if order == -1:
            qp = MESH.FEMLISTS[space]['TET']['qp_we_K'][0]; 
            we = MESH.FEMLISTS[space]['TET']['qp_we_K'][1]; nqp = len(we)
        
        ellmatsB = npy.zeros((nqp*nt,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[:nt*nqp].reshape(nt,nqp).T.flatten(),(lphi,1)).T
                
        for j in range(lphi):
            for i in range(nqp):
                divphii = divphi[j](qp[0,i],qp[1,i],qp[2,i])
                detA = MESH.detA(qp[0,i],qp[1,i],qp[2,i])
                ellmatsB[i*nt:(i+1)*nt,j] = 1/detA*divphii*DIRECTION_DOF[:,j]
                
        B = sparse(im,jm,ellmatsB,sizeM,nqp*nt)
        return B
    


# ACHTUNG! Nur outer faces nach außen orietiert!
# def assembleB3(MESH, space, matrix, shape, order = -1, faces = npy.empty(0)):
def assembleB3(MESH, space, matrix, shape, order = -1):
    
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH,space)
    
    p = MESH.p
    f = MESH.f;
    
    phi = MESH.FEMLISTS[space]['B']['phi']; lphi = len(phi)
    
    # if faces.size == 0:
    #     f = MESH.f
    # else:
    #     f = MESH.FacesToVertices[faces,:]
        
    nf = f.shape[0]
    LIST_DOF = MESH.FEMLISTS[space]['B']['LIST_DOF']
    
    if order != -1:
        qp,we = quadrature.dunavant(order); nqp = len(we)
    
    #####################################################################################
    # Mass matrix (over the edge)
    #####################################################################################

    if matrix == 'M':
        if order == -1:
            qp = MESH.FEMLISTS[space]['B']['qp_we_B'][0]; 
            we = MESH.FEMLISTS[space]['B']['qp_we_B'][1]; nqp = len(we)
        
        ellmatsB = npy.zeros((nqp*nf,lphi))
        
        im = npy.tile(LIST_DOF,(nqp,1))
        jm = npy.tile(npy.c_[:nf*nqp].reshape(nf,nqp).T.flatten(),(lphi,1)).T
        
        for j in range(lphi):
            for i in range(nqp):
                detB = MESH.detB(qp[0,i],qp[1,i])
                print(j,i,abs(detB).max(),abs(detB).min())
                ellmatsB[i*nf:(i+1)*nf,j] = 1/npy.abs(detB)*phi[j](qp[0,i],qp[1,i])
        
        print(im.shape,jm.shape,ellmatsB.shape,nqp,nf,shape)
        B = sparse(im,jm,ellmatsB,shape[0],nqp*nf)
        return B
    
    

# def assembleE(MESH,space,matrix,order=1):
    
#     if not space in MESH.FEMLISTS.keys():
#         spaceInfo(MESH,space)
    
#     p = MESH.p; 
#     t = MESH.t; nt = MESH.nt
    
#     phi =  MESH.FEMLISTS[space]['TET']['phi']; lphi = len(phi)
#     sizeM = MESH.FEMLISTS[space]['TET']['sizeM']
        
#     LIST_DOF = MESH.FEMLISTS[space]['TET']['LIST_DOF']
#     DIRECTION_DOF = MESH.FEMLISTS[space]['TET']['DIRECTION_DOF']
    
#     qp,we = quadrature.one_d(order); nqp = len(we)
    
#     # on edge2: (0,0) -> (1,0)
#     qp2 = npy.c_[1-qp,0*qp].T
#     # on edge1: (0,0) -> (0,1)
#     qp1 = npy.c_[0*qp,qp].T
#     # on edge0: (0,1) -> (1,0)
#     qp0 = npy.c_[qp,1-qp].T
    
#     #####################################################################################
#     # Mappings
#     #####################################################################################
    
#     t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]
#     A00 = p[t1,0]-p[t0,0]; A01 = p[t2,0]-p[t0,0];
#     A10 = p[t1,1]-p[t0,1]; A11 = p[t2,1]-p[t0,1];
#     detA = A00*A11-A01*A10
    
#     #####################################################################################
#     # Mass matrix (over the edge)
#     #####################################################################################
    
#     if matrix == 'M':
        
#         ellmatsB0 = npy.zeros((nqp*nt,lphi))
#         ellmatsB1 = npy.zeros((nqp*nt,lphi))
#         ellmatsB2 = npy.zeros((nqp*nt,lphi))
        
#         im = npy.tile(LIST_DOF,(nqp,1))
        
#         ind0 = npy.argwhere(MESH.EdgeDirectionTET[:,0]==-1)
#         ind1 = npy.argwhere(MESH.EdgeDirectionTET[:,1]==-1)
#         ind2 = npy.argwhere(MESH.EdgeDirectionTET[:,2]==-1)
        
#         indices = npy.tile(npy.arange(nqp),(nt,1))
        
#         indices0 = indices.copy(); indices1 = indices.copy(); indices2 = indices.copy()
        
#         indices0[ind0,:] = indices0[ind0,::-1]; 
#         indices1[ind1,:] = indices1[ind1,::-1]; 
#         indices2[ind2,:] = indices2[ind2,::-1];
        
#         jm0 = npy.tile(npy.tile(MESH.TriangleToEdges[:,0]*nqp,nqp) + indices0.T.flatten(),(lphi,1)).T
#         jm1 = npy.tile(npy.tile(MESH.TriangleToEdges[:,1]*nqp,nqp) + indices1.T.flatten(),(lphi,1)).T
#         jm2 = npy.tile(npy.tile(MESH.TriangleToEdges[:,2]*nqp,nqp) + indices2.T.flatten(),(lphi,1)).T

#         for j in range(lphi):
#             for i in range(nqp):
#                 phii0 = phi[j](qp0[0,i],qp0[1,i])
#                 phii1 = phi[j](qp1[0,i],qp1[1,i])
#                 phii2 = phi[j](qp2[0,i],qp2[1,i])
                
#                 ellmatsB0[i*nt:(i+1)*nt,j] = 1/detA*( A11*phii0[0] -A10*phii0[1])*DIRECTION_DOF[:,j]*MESH.tangent0[:,0]+\
#                                              1/detA*(-A01*phii0[0] +A00*phii0[1])*DIRECTION_DOF[:,j]*MESH.tangent1[:,0]
#                 ellmatsB1[i*nt:(i+1)*nt,j] = 1/detA*( A11*phii1[0] -A10*phii1[1])*DIRECTION_DOF[:,j]*MESH.tangent0[:,1]+\
#                                              1/detA*(-A01*phii1[0] +A00*phii1[1])*DIRECTION_DOF[:,j]*MESH.tangent1[:,1]
#                 ellmatsB2[i*nt:(i+1)*nt,j] = 1/detA*( A11*phii2[0] -A10*phii2[1])*DIRECTION_DOF[:,j]*MESH.tangent0[:,2]+\
#                                              1/detA*(-A01*phii2[0] +A00*phii2[1])*DIRECTION_DOF[:,j]*MESH.tangent1[:,2]
        
#         ellmatsB0 = ellmatsB0*(npy.abs(ellmatsB0)>1e-12)
#         ellmatsB1 = ellmatsB1*(npy.abs(ellmatsB1)>1e-12)
#         ellmatsB2 = ellmatsB2*(npy.abs(ellmatsB2)>1e-12)
        
#         B0 = sparse(im,jm0,ellmatsB0,sizeM,nqp*MESH.NoEdges)
#         B1 = sparse(im,jm1,ellmatsB1,sizeM,nqp*MESH.NoEdges)
#         B2 = sparse(im,jm2,ellmatsB2,sizeM,nqp*MESH.NoEdges)
        
#         B0.eliminate_zeros()
#         B1.eliminate_zeros()
#         B2.eliminate_zeros()
        
#         return B0,B1,B2

# def assembleR3(MESH, space, faces = '', listDOF = npy.empty(0)):
    
#     if not space in MESH.FEMLISTS.keys():
#         spaceInfo(MESH,space)
        
#     if type(faces) == str:
#         if faces == '':
#             ind_faces = MESH.BoundaryEdges_Region
#         else:
#             ind_faces = getIndices(MESH.regions_2d,faces)
#     else:
#         if MESH.regions_2d == []:
#             ind_faces = edges
#         else:
#             ind_faces = getIndices(MESH.regions_2d,faces)
    
#     indices = npy.in1d(MESH.BoundaryEdges_Region,ind_edges)
#     sizeM = MESH.FEMLISTS[space]['TET']['sizeM']
    
#     LIST_DOF  = npy.unique(MESH.FEMLISTS[space]['B']['LIST_DOF'][indices])
#     LIST_DOF2 = npy.setdiff1d(npy.arange(sizeM),LIST_DOF)
    
#     D = sp.eye(sizeM, format = 'csc')
    
#     if listDOF.size > 0:
#         LIST_DOF = listDOF
    
#     R1 = D[:,LIST_DOF]
#     R2 = D[:,LIST_DOF2]
    
#     return R1.T.tocsc(),R2.T.tocsc()
    
def sparse(i, j, v, m, n):
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape = (m, n))