
from scipy import sparse as sp
import numpy as npy
from .. import quadrature
# import numba as nb


# @profile
def assemble(MESH,order):
    
    p = MESH.p;
    t = MESH.t; nt = t.shape[0]
    
    qp,we = quadrature.dunavant(order); nqp = len(we)

    #####################################################################################
    # Mappings
    #####################################################################################

    t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]
    A00 = p[t1,0]-p[t0,0]; A01 = p[t2,0]-p[t0,0]
    A10 = p[t1,1]-p[t0,1]; A11 = p[t2,1]-p[t0,1]
    detA = A00*A11-A01*A10
    
    #####################################################################################
    
    ellmatsD = npy.zeros((nqp*nt))
    iD = npy.r_[0:nqp*nt].reshape(nt,nqp).T
    
    for i in range(nqp):
        ellmatsD[i*nt:(i+1)*nt] = 1/2*npy.abs(detA)*we[i]
    
    D = sparse(iD,iD,ellmatsD,nqp*nt,nqp*nt)
    return D


def assembleB(MESH,order):
    
    p = MESH.p;
    e = MESH.e; ne = e.shape[0]
    
    qp,we = quadrature.one_d(order); nqp = len(we)
            
    #####################################################################################
    # Mappings
    #####################################################################################
        
    e0 = e[:,0]; e1 = e[:,1]
    A0 =  p[e1,0]-p[e0,0]; A1 =  p[e1,1]-p[e0,1]
    detA = npy.sqrt(A0**2+A1**2)
    
    #####################################################################################

    ellmatsD = npy.zeros((nqp*ne))
    
    iD = npy.r_[0:nqp*ne].reshape(ne,nqp).T
    
    for i in range(nqp):
        ellmatsD[i*ne:(i+1)*ne] = npy.abs(detA)*we[i]
    
    D = sparse(iD,iD,ellmatsD,nqp*ne,nqp*ne)
    return D


def assembleE(MESH,order):
    
    p = MESH.p;
    e = MESH.EdgesToVertices; ne = e.shape[0]
    
    qp,we = quadrature.one_d(order); nqp = len(we)
            
    #####################################################################################
    # Mappings
    #####################################################################################
        
    e0 = e[:,0]; e1 = e[:,1]
    A0 =  p[e1,0]-p[e0,0]; A1 =  p[e1,1]-p[e0,1]
    detA = npy.sqrt(A0**2+A1**2)
    
    #####################################################################################

    ellmatsD = npy.zeros((nqp*ne))
    iD = npy.r_[0:nqp*ne].reshape(ne,nqp).T
    
    for i in range(nqp):
        ellmatsD[i*ne:(i+1)*ne] = npy.abs(detA)*we[i]
    
    D = sparse(iD,iD,ellmatsD,nqp*ne,nqp*ne)
    return D

def evaluate(MESH, order, coeff = lambda x,y : 1+0*x*y, regions = '', indices = npy.empty(0)):
    
    if indices.size == 0:
        if regions == '':
            ind_regions = MESH.t[:,-1]
        elif isinstance(regions, npy.ndarray):
            ind_regions = regions;
        else:
            ind_regions = MESH.getIndices2d(MESH.regions_2d,regions)
        indices = npy.in1d(MESH.t[:,-1],ind_regions)
    
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

def evaluateB(MESH, order, coeff = lambda x,y : 1+0*x*y, edges = '', like = 0):
    
    # if edges.size == 0:
    #     edges = MESH.Boundary_Region
    
    # indices = npy.argwhere(npy.in1d(MESH.Boundary_Region,edges))[:,0]
    # indices = npy.in1d(MESH.Boundary_Region,edges)
    
    # if edges == '':
    #     ind_edges = MESH.Boundary_Region
    # else:
    #     ind_edges = MESH.getIndices2d(MESH.regions_1d,edges)
    # indices = npy.in1d(MESH.Boundary_Region,ind_edges)
    
    
    if type(edges) == str:
        if edges == '':
            ind_edges = MESH.Boundary_Region
        else:
            ind_edges = MESH.getIndices2d(MESH.regions_1d,edges)
    else:
        if MESH.regions_1d == []:
            ind_edges = edges
        
    indices = npy.in1d(MESH.Boundary_Region,ind_edges)
    
    
    p = MESH.p;    
    e = MESH.e[indices,:]; ne = e.shape[0]
    
    # print(ind_edges,MESH.e.shape,e.shape)
    
    #####################################################################################
    # Mappings
    #####################################################################################

    e0 = e[:,0]; e1 = e[:,1]
    A0 = p[e1,0]-p[e0,0]; A1 = p[e1,1]-p[e0,1]
    
    #####################################################################################
    # Custom config matrix
    #####################################################################################
    
    qp,we = quadrature.one_d(order); nqp = len(we)
    ellmatsD = npy.zeros((nqp*ne))
    
    iD = npy.r_[0:nqp*MESH.ne].reshape(MESH.ne,nqp).T
    iD = iD[:,indices]
    
    d = npy.zeros(nqp*MESH.ne)
    
    for i in range(nqp):
        qpT_i_1 = A0*qp[i] + p[e0,0]
        qpT_i_2 = A1*qp[i] + p[e0,1]
        ellmatsD[i*ne:(i+1)*ne] = coeff(qpT_i_1,qpT_i_2)
    
    # D = sparse(iD,iD,ellmatsD,nqp*MESH.ne,nqp*MESH.ne)
    d[iD.flatten()] = ellmatsD
    
    if like == 0:
        return sp.diags(d)
    
    if like == 1:
        return d


def evaluateE(MESH, order, coeff = lambda x,y : 1+0*x*y, edges = '', like = 0):
    
    # if edges.size == 0:
    #     edges = MESH.Boundary_Region
    
    # # indices = npy.argwhere(npy.in1d(MESH.Boundary_Region,edges))[:,0]
    # indices = npy.in1d(MESH.Boundary_Region,edges)
    
    
    if edges == '':
        ind_edges = MESH.Boundary_Region
    else:
        if MESH.regions_1d == []:
            ind_edges = edges
        else:
            ind_edges = MESH.getIndices2d(MESH.regions_1d,edges)
    indices = npy.in1d(MESH.EdgesToVertices[:,-1],ind_edges)
    
    
    
    p = MESH.p;    
    # e = MESH.e[indices,:]; ne = e.shape[0]
    # print(ind_edges)
    # print(indices)
    e = MESH.EdgesToVertices[ind_edges,:]; ne = e.shape[0]
    ne_full = MESH.EdgesToVertices.shape[0]
    
    
    #####################################################################################
    # Mappings
    #####################################################################################
    
    e0 = e[:,0]; e1 = e[:,1]
    A0 = p[e1,0]-p[e0,0]; A1 = p[e1,1]-p[e0,1]
    
    #####################################################################################
    # Custom config matrix
    #####################################################################################
    
    qp,we = quadrature.one_d(order); nqp = len(we)
    ellmatsD = npy.zeros((nqp*ne))
    
    # iD = npy.r_[0:nqp*MESH.ne].reshape(MESH.ne,nqp).T
    iD = npy.r_[0:nqp*ne_full].reshape(ne_full,nqp).T
    iD = iD[:,indices]
    
    
    d = npy.zeros(nqp*ne_full)
    
    for i in range(nqp):
        qpT_i_1 = A0*qp[i] + p[e0,0]
        qpT_i_2 = A1*qp[i] + p[e0,1]
        ellmatsD[i*ne:(i+1)*ne] = coeff(qpT_i_1,qpT_i_2)
    
    # print(ellmatsD.shape,iD.flatten().shape)
    
    # D = sparse(iD,iD,ellmatsD,nqp*MESH.ne,nqp*MESH.ne)
    d[iD.flatten()] = ellmatsD
    
    if like == 0:
        return sp.diags(d)
    
    if like == 1:
        return d

# def reduceEssential(MESH, order, edges = npy.empty(0)):
    
#     if edges.size == 0:
#         edges = MESH.Boundary_Region
    
#     indices = npy.argwhere(npy.in1d(MESH.Boundary_Region,edges))[:,0]

#     p = MESH.p;    
#     e = MESH.e[indices,:]; ne = e.shape[0]
    
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

#     iD = npy.r_[0:nqp*MESH.ne].reshape(MESH.ne,nqp).T
#     iD = iD[:,indices]

#     for i in range(nqp):
#         qpT_i_1 = A0*qp[i] + p[e0,0]
#         qpT_i_2 = A1*qp[i] + p[e0,1]
#         ellmatsD[i*ne:(i+1)*ne] = coeff(qpT_i_1,qpT_i_2)

#     D = sparse(iD,iD,ellmatsD,nqp*MESH.ne,nqp*MESH.ne)
#     return D


def sparse(i, j, v, m, n):
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape = (m, n))