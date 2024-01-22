
from scipy import sparse as sp
import numpy as npy
from .. import quadrature
from ..tools import getIndices
# import numba as nb


# @profile
def assemble3(MESH,order):
    
    p = MESH.p
    t = MESH.t; nt = t.shape[0]
    
    qp,we = quadrature.keast(order); nqp = len(we)
    
    #####################################################################################
    
    ellmatsD = npy.zeros((nqp*nt))
    iD = npy.r_[:nqp*nt].reshape(nt,nqp).T
    
    for i in range(nqp):
        detA = MESH.detA(qp[0,i],qp[1,i],qp[2,i])
        ellmatsD[i*nt:(i+1)*nt] = 1/6*npy.abs(detA)*we[i]
    
    D = sparse(iD,iD,ellmatsD,nqp*nt,nqp*nt)
    return D
def assembleN3(MESH,order):

    p = MESH.p
    f = MESH.f; nf = f.shape[0]

    qp,we = quadrature.dunavant(order); nqp = len(we)

    #####################################################################################

    ellmatsN1 = npy.zeros((nqp*nf))
    ellmatsN2 = npy.zeros((nqp*nf))
    ellmatsN3 = npy.zeros((nqp*nf))
    iN = npy.r_[:nqp*nf].reshape(nf,nqp).T

    for i in range(nqp):
        ellmatsN1[i*nf:(i+1)*nf] = MESH.normals[:,0]
        ellmatsN2[i*nf:(i+1)*nf] = MESH.normals[:,1]
        ellmatsN3[i*nf:(i+1)*nf] = MESH.normals[:,2]

    N1 = sparse(iN,iN,ellmatsN1,nqp*nf,nqp*nf).diagonal()
    N2 = sparse(iN,iN,ellmatsN2,nqp*nf,nqp*nf).diagonal()
    N3 = sparse(iN,iN,ellmatsN3,nqp*nf,nqp*nf).diagonal()
    return N1,N2,N3


def assembleB3(MESH,order):
    
    p = MESH.p
    f = MESH.f; nf = f.shape[0]
    
    qp,we = quadrature.dunavant(order); nqp = len(we)
    
    #####################################################################################

    ellmatsD = npy.zeros((nqp*nf))
    
    iD = npy.r_[:nqp*nf].reshape(nf,nqp).T
    
    for i in range(nqp):
        detB = MESH.detB(qp[0,i],qp[1,i])
        ellmatsD[i*nf:(i+1)*nf] = 1/2*npy.abs(detB)*we[i]
    
    D = sparse(iD,iD,ellmatsD,nqp*nf,nqp*nf)
    return D


# def assembleE(MESH,order):
    
#     p = MESH.p;
#     e = MESH.EdgesToVertices; ne = e.shape[0]
    
#     qp,we = quadrature.one_d(order); nqp = len(we)
            
#     #####################################################################################
#     # Mappings
#     #####################################################################################
        
#     e0 = e[:,0]; e1 = e[:,1]
#     A0 =  p[e1,0]-p[e0,0]; A1 =  p[e1,1]-p[e0,1]
#     detA = npy.sqrt(A0**2+A1**2)
    
#     #####################################################################################

#     ellmatsD = npy.zeros((nqp*ne))
#     iD = npy.r_[0:nqp*ne].reshape(ne,nqp).T
    
#     for i in range(nqp):
#         ellmatsD[i*ne:(i+1)*ne] = npy.abs(detA)*we[i]
    
#     D = sparse(iD,iD,ellmatsD,nqp*ne,nqp*ne)
#     return D

def evaluate3(MESH, order, coeff = lambda x,y,z : 1+0*x*y*z, regions = '', indices = npy.empty(0)):
    
    if indices.size == 0:
        if regions == '':
            ind_regions = MESH.t[:,-1]
        elif isinstance(regions, npy.ndarray):
            ind_regions = regions;
        else:
            ind_regions = getIndices(MESH.regions_3d,regions)
        indices = npy.in1d(MESH.t[:,-1],ind_regions)
    
    p = MESH.p;
    t = MESH.t[indices,:]; nt = t.shape[0]
    
    #####################################################################################
    # Custom config matrix
    #####################################################################################
    
    qp,we = quadrature.keast(order); nqp = len(we)
    ellmatsD = npy.zeros((nqp*nt))
    
    iD = npy.r_[:nqp*MESH.nt].reshape(MESH.nt,nqp).T
    iD = iD[:,indices]
    
    for i in range(nqp):
        qpT_i_1 = MESH.Fx(qp[0,i],qp[1,i],qp[2,i])[indices]
        qpT_i_2 = MESH.Fy(qp[0,i],qp[1,i],qp[2,i])[indices]
        qpT_i_3 = MESH.Fz(qp[0,i],qp[1,i],qp[2,i])[indices]
        
        ellmatsD[i*nt:(i+1)*nt] = coeff(qpT_i_1, qpT_i_2, qpT_i_3)
    
    D = sparse(iD,iD,ellmatsD,nqp*MESH.nt,nqp*MESH.nt)
    return D



def evaluateB3(MESH, order, coeff = lambda x,y,z : 1+0*x*y*z, faces = '', like = 0):
    
    
    if type(faces) == str:
        if faces == '':
            ind_faces = MESH.Boundary_Faces
        else:
            ind_faces = getIndices(MESH.regions_2d,faces)
    else:
        if MESH.regions_2d == []:
            ind_faces = faces
        
    indices = npy.in1d(MESH.BoundaryFaces_Region,ind_faces)
    
    p = MESH.p
    f = MESH.f[indices,:]; nf = f.shape[0]
    
    #####################################################################################
    # Custom config matrix
    #####################################################################################
    
    qp,we = quadrature.dunavant(order); nqp = len(we)
    ellmatsD = npy.zeros((nqp*nf))
    
    iD = npy.r_[:nqp*MESH.nf].reshape(MESH.nf,nqp).T
    iD = iD[:,indices]
    
    d = npy.zeros(nqp*MESH.nf)
    
    for i in range(nqp):
        # This is also weird!
        qpT_i_1 = MESH.Bx(qp[0,i],qp[1,i])[indices]
        qpT_i_2 = MESH.By(qp[0,i],qp[1,i])[indices]
        qpT_i_3 = MESH.Bz(qp[0,i],qp[1,i])[indices]
        ellmatsD[i*nf:(i+1)*nf] = coeff(qpT_i_1,qpT_i_2,qpT_i_3)
    
    d[iD.flatten()] = ellmatsD
    
    if like == 0:
        return sp.diags(d)
    
    if like == 1:
        return d


# def evaluateE(MESH, order, coeff = lambda x,y : 1+0*x*y, edges = '', like = 0):
    
#     # if edges.size == 0:
#     #     edges = MESH.Boundary_Region
    
#     # # indices = npy.argwhere(npy.in1d(MESH.Boundary_Region,edges))[:,0]
#     # indices = npy.in1d(MESH.Boundary_Region,edges)
    
    
#     if edges == '':
#         ind_edges = MESH.Boundary_Region
#     else:
#         if MESH.regions_1d == []:
#             ind_edges = edges
#         else:
#             ind_edges = MESH.getIndices2d(MESH.regions_1d,edges)
#     indices = npy.in1d(MESH.EdgesToVertices[:,-1],ind_edges)
    
    
    
#     p = MESH.p;    
#     # e = MESH.e[indices,:]; ne = e.shape[0]
#     # print(ind_edges)
#     # print(indices)
#     e = MESH.EdgesToVertices[ind_edges,:]; ne = e.shape[0]
#     ne_full = MESH.EdgesToVertices.shape[0]
    
    
#     #####################################################################################
#     # Mappings
#     #####################################################################################
    
#     e0 = e[:,0]; e1 = e[:,1]
#     A0 = p[e1,0]-p[e0,0]; A1 = p[e1,1]-p[e0,1]
    
#     #####################################################################################
#     # Custom config matrix
#     #####################################################################################
    
#     qp,we = quadrature.one_d(order); nqp = len(we)
#     ellmatsD = npy.zeros((nqp*ne))
    
#     # iD = npy.r_[0:nqp*MESH.ne].reshape(MESH.ne,nqp).T
#     iD = npy.r_[0:nqp*ne_full].reshape(ne_full,nqp).T
#     iD = iD[:,indices]
    
    
#     d = npy.zeros(nqp*ne_full)
    
#     for i in range(nqp):
#         qpT_i_1 = A0*qp[i] + p[e0,0]
#         qpT_i_2 = A1*qp[i] + p[e0,1]
#         ellmatsD[i*ne:(i+1)*ne] = coeff(qpT_i_1,qpT_i_2)
    
#     # print(ellmatsD.shape,iD.flatten().shape)
    
#     # D = sparse(iD,iD,ellmatsD,nqp*MESH.ne,nqp*MESH.ne)
#     d[iD.flatten()] = ellmatsD
    
#     if like == 0:
#         return sp.diags(d)
    
#     if like == 1:
#         return d



def sparse(i, j, v, m, n):
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape = (m, n))