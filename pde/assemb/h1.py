
from re import I
from .. import quadrature
import numpy as npy
from scipy import sparse as sp
# import matplotlib.pyplot as plt
import time

print('lele')

def get_info(MESH,space):
    
    def get_info_trig(space):
        
        INFO = {}
        INFO['TRIG'] = {}
                    
        if space == 'P1':
            INFO['TRIG']['space'] = 'P1'
            INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
            INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
            INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
            
        return INFO
    
    def get_info_quad(space):
        INFO = {}
        INFO['QUAD'] = {}
        
        if space == 'Q1':
            INFO['QUAD']['space'] = 'Q1'
            INFO['QUAD']['qp_we_M'] = quadrature.quadrule(order = 3)
            INFO['QUAD']['qp_we_Mh'] = quadrature.quadrule(order = 1)
            INFO['QUAD']['qp_we_K'] = quadrature.quadrule(order = 3)
            
        return INFO
    
    INFO = {}
    
    ###########################################################################
    if space == 'P1-Q1': 
        INFO = INFO | get_info_trig('P1')
        INFO = INFO | get_info_quad('Q1')
        INFO['sizeM'] = MESH.np
    ###########################################################################
    
    if space == 'P1b':
        INFO['qp_we_B'] = quadrature.one_d(order = 2)
    
    return INFO









def M(MESH,BASIS,LISTS,Dict):
    time_elapsed = time.time()
    space = Dict.get('space')
    matrix = Dict.get('matrix')
    
    MAT = {}; MAT[space] = {}
    MAT_TRIG = {}; MAT_QUAD = {}
    
    t = MESH.t; q = MESH.q
    
    INFO = get_info(MESH,space)

    if t.shape[0] != 0: MAT_TRIG = h1_trig(MESH,BASIS,LISTS,INFO,Dict)
    if q.shape[0] != 0: MAT_QUAD = h1_quad(MESH,BASIS,LISTS,INFO,Dict)

    if t.shape[0] == 0: MAT[space] = MAT_QUAD
    if q.shape[0] == 0: MAT[space] = MAT_TRIG
    
    # MAT_BOUNDARY = h1_boundary(MESH,BASIS,LISTS,INFO)

    if t.shape[0] != 0 and q.shape[0] != 0:
        MAT_TRIG_Set = set(MAT_TRIG)
        MAT_QUAD_Set = set(MAT_QUAD)
        
        for field in MAT_TRIG_Set.intersection(MAT_QUAD_Set):
            MAT[space][field] = MAT_TRIG[field] + MAT_QUAD[field]
            
    elapsed = time.time()-time_elapsed
    print('Assembling ' + space + ' took ' + str(elapsed)[0:5] + ' seconds.')
    return MAT

def h1_trig(MESH,BASIS,LISTS,INFO,Dict):

    p = MESH.p;
    t = MESH.t; nt = MESH.nt
    
    spaceTrig = INFO['TRIG']['space'];
    qp_M = INFO['TRIG']['qp_we_M'][0]; we_M = INFO['TRIG']['qp_we_M'][1]
    qp_K = INFO['TRIG']['qp_we_K'][0]; we_K = INFO['TRIG']['qp_we_K'][1]

    phi_H1 = BASIS[spaceTrig]['TRIG']['phi']; lphi_H1 = len(phi_H1)
    dphi_H1 = BASIS[spaceTrig]['TRIG']['dphi']; ldphi_H1 = len(dphi_H1)

    H1_LIST_DOF = LISTS[spaceTrig]['TRIG']['LIST_DOF']

    ellmatsM_H1 = npy.zeros((nt,lphi_H1*lphi_H1))
    ellmatsM_H1_LUMPED = npy.zeros((nt,lphi_H1*lphi_H1))
    
    ellmatsKx_H1 = npy.zeros((nt,ldphi_H1*ldphi_H1))
    ellmatsKy_H1 = npy.zeros((nt,ldphi_H1*ldphi_H1))

    phiix_H1 = npy.zeros((nt,lphi_H1))
    phiiy_H1 = npy.zeros((nt,lphi_H1))
    phii_H1 = npy.zeros((nt,lphi_H1)) 

    MAT = {}

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

    for i in range(len(we_M)):
        for j in range(lphi_H1):
            phii_H1[:,j] = phi_H1[j](qp_M[0,i],qp_M[1,i])

        ellmatsM_H1 = ellmatsM_H1 + 1/2*we_M[i]*(assem_ellmats(phii_H1,phii_H1))*npy.abs(detA)[:,None]

    im_H1,jm_H1 = create_indices(H1_LIST_DOF,H1_LIST_DOF)
    ellmatsM_H1[abs(ellmatsM_H1)<1e-14] = 0
    MAT['M'] = sparse(im_H1,jm_H1,ellmatsM_H1,INFO['sizeM'],INFO['sizeM'])
    MAT['M'].eliminate_zeros()

    #####################################################################################
    # Stiffness matrix
    #####################################################################################

    for i in range(len(we_K)):
        for j in range(lphi_H1):
            dphii_H1 = dphi_H1[j](qp_K[0,i],qp_K[1,i])
            phiix_H1[:,j] = 1/detA*( A11*dphii_H1[0] -A10*dphii_H1[1])
            phiiy_H1[:,j] = 1/detA*(-A01*dphii_H1[0] +A00*dphii_H1[1])

        ellmatsKx_H1 = ellmatsKx_H1 + 1/2*we_K[i]*assem_ellmats(phiix_H1,phiix_H1)*npy.abs(detA)[:,None]
        ellmatsKy_H1 = ellmatsKy_H1 + 1/2*we_K[i]*assem_ellmats(phiiy_H1,phiiy_H1)*npy.abs(detA)[:,None]

    ellmatsKx_H1[abs(ellmatsKx_H1)<1e-14] = 0
    ellmatsKy_H1[abs(ellmatsKy_H1)<1e-14] = 0
    
    MAT['Kx'] = sparse(im_H1,jm_H1,ellmatsKx_H1,INFO['sizeM'],INFO['sizeM']); MAT['Kx'].eliminate_zeros()
    MAT['Ky'] = sparse(im_H1,jm_H1,ellmatsKy_H1,INFO['sizeM'],INFO['sizeM']); MAT['Ky'].eliminate_zeros()
    
    MAT['M'].eliminate_zeros()
        
    #####################################################################################
    # Lumped mass matrix
    #####################################################################################

    # if 'qp_Mh' in locals():
    if 'qp_we_Mh' in INFO['TRIG']:
        qp_Mh = INFO['TRIG']['qp_we_Mh'][0]; we_Mh = INFO['TRIG']['qp_we_Mh'][1]
        for i in range(len(we_Mh)):
            for j in range(lphi_H1):
                phii_H1[:,j] = phi_H1[j](qp_Mh[0,i],qp_Mh[1,i])

            ellmatsM_H1_LUMPED = ellmatsM_H1_LUMPED + 1/2*we_M[i]*(assem_ellmats(phii_H1,phii_H1))*npy.abs(detA)[:,None]
        
        ellmatsM_H1_LUMPED[abs(ellmatsM_H1_LUMPED)<1e-14] = 0
        MAT['Mh'] = sparse(im_H1,jm_H1,ellmatsM_H1_LUMPED,INFO['sizeM'],INFO['sizeM'])
        MAT['Mh'].eliminate_zeros()
    return MAT





def h1_quad(MESH,BASIS,LISTS,INFO):
    p = MESH.p; q = MESH.q; nq = MESH.nq
    
    spaceQuad = INFO['QUAD']['space'];
    qp_M = INFO['QUAD']['qp_we_M'][0]; we_M = INFO['QUAD']['qp_we_M'][1]
    qp_K = INFO['QUAD']['qp_we_K'][0]; we_K = INFO['QUAD']['qp_we_K'][1]

    phi_H1 = BASIS[spaceQuad]['QUAD']['phi']; lphi_H1 = len(phi_H1)
    dphi_H1 = BASIS[spaceQuad]['QUAD']['dphi']; ldphi_H1 = len(dphi_H1)

    H1_LIST_DOF = LISTS[spaceQuad]['QUAD']['LIST_DOF'] + INFO['list_shift']

    ellmatsM_H1 = npy.zeros((nq,lphi_H1*lphi_H1))
    ellmatsM_H1_LUMPED = npy.zeros((nq,lphi_H1*lphi_H1))
    
    ellmatsKx_H1 = npy.zeros((nq,ldphi_H1*ldphi_H1))
    ellmatsKy_H1 = npy.zeros((nq,ldphi_H1*ldphi_H1))

    phiix_H1 = npy.zeros((nq,lphi_H1))
    phiiy_H1 = npy.zeros((nq,lphi_H1))
    phii_H1 = npy.zeros((nq,lphi_H1))
    
    MAT = {}

    #####################################################################################
    # Mappings
    #####################################################################################
    
    q0 = q[:,0]; q1 = q[:,1]; q2 = q[:,2]; q3 = q[:,3]
    
    B00 = p[q1,0]-p[q0,0]; B01 = p[q3,0]-p[q0,0]
    B10 = p[q1,1]-p[q0,1]; B11 = p[q3,1]-p[q0,1]
    detB = abs(B00*B11-B01*B10)
    
    C00 = p[q2,0]-p[q3,0]; C01 = p[q3,0]-p[q0,0]
    C10 = p[q2,1]-p[q3,1]; C11 = p[q3,1]-p[q0,1]
    detC = abs(C00*C11-C01*C10)
    
    D00 = p[q1,0]-p[q0,0]; D01 = p[q2,0]-p[q1,0]
    D10 = p[q1,1]-p[q0,1]; D11 = p[q2,1]-p[q1,1]
    detD = abs(D00*D11-D01*D10)
    
    detQ = lambda x,y: detB + (detD-detB)*x + (detC-detB)*y
    
    R0 = p[q2,0]-p[q3,0]-p[q1,0]+p[q0,0]
    R1 = p[q2,1]-p[q3,1]-p[q1,1]+p[q0,1]
    
    Q00 = lambda x,y : B00+R0*y; Q01 = lambda x,y : B01+R1*x
    Q10 = lambda x,y : B10+R0*y; Q11 = lambda x,y : B11+R1*x # not 100% sure on this one ...    
    
    #####################################################################################
    # Mass matrix
    #####################################################################################
    
    MAT = {}
    
    for i in range(len(we_M)):
        detQi = detQ(qp_M[0,i],qp_M[1,i])        
        for j in range(lphi_H1):
            phii_H1[:,j] = phi_H1[j](qp_M[0,i],qp_M[1,i])

        ellmatsM_H1 = ellmatsM_H1 + we_M[i]*(assem_ellmats(phii_H1,phii_H1))*npy.abs(detQi)[:,None]

    im_H1,jm_H1 = create_indices(H1_LIST_DOF,H1_LIST_DOF)
    ellmatsM_H1[abs(ellmatsM_H1)<1e-14] = 0
    MAT['M'] = sparse(im_H1,jm_H1,ellmatsM_H1,INFO['sizeM'],INFO['sizeM'])
    MAT['M'].eliminate_zeros()

    #####################################################################################
    # Stiffness matrix
    #####################################################################################

    for i in range(len(we_K)):
        detQi = detQ(qp_K[0,i],qp_K[1,i])
        Q00i = Q00(qp_K[0,i],qp_K[1,i]); Q10i = Q10(qp_K[0,i],qp_K[1,i])
        Q01i = Q01(qp_K[0,i],qp_K[1,i]); Q11i = Q11(qp_K[0,i],qp_K[1,i])

        for j in range(lphi_H1):
            dphii_H1 = dphi_H1[j](qp_K[0,i],qp_K[1,i])
            phiix_H1[:,j] = 1/detQi*( Q11i*dphii_H1[0] -Q10i*dphii_H1[1])
            phiiy_H1[:,j] = 1/detQi*(-Q01i*dphii_H1[0] +Q00i*dphii_H1[1])

        ellmatsKx_H1 = ellmatsKx_H1 + we_K[i]*assem_ellmats(phiix_H1,phiix_H1)*npy.abs(detQi)[:,None]
        ellmatsKy_H1 = ellmatsKy_H1 + we_K[i]*assem_ellmats(phiiy_H1,phiiy_H1)*npy.abs(detQi)[:,None]
        
    ellmatsKx_H1[abs(ellmatsKx_H1)<1e-14] = 0
    ellmatsKy_H1[abs(ellmatsKy_H1)<1e-14] = 0
   
    MAT['Kx'] = sparse(im_H1,jm_H1,ellmatsKx_H1,INFO['sizeM'],INFO['sizeM']); MAT['Kx'].eliminate_zeros()
    MAT['Ky'] = sparse(im_H1,jm_H1,ellmatsKy_H1,INFO['sizeM'],INFO['sizeM']); MAT['Ky'].eliminate_zeros()

    #####################################################################################
    # Lumped mass matrix
    #####################################################################################

    # if 'qp_Mh' in locals():
    if 'qp_we_Mh' in INFO['QUAD']:
        qp_Mh = INFO['QUAD']['qp_we_Mh'][0]; we_Mh = INFO['QUAD']['qp_we_Mh'][1]
        for i in range(len(we_Mh)):
            detQi = detQ(qp_Mh[0,i],qp_Mh[1,i])        
            for j in range(lphi_H1):
                phii_H1[:,j] = phi_H1[j](qp_Mh[0,i],qp_Mh[1,i])

            ellmatsM_H1_LUMPED = ellmatsM_H1_LUMPED + we_Mh[i]*(assem_ellmats(phii_H1,phii_H1))*npy.abs(detQi)[:,None]
             
        ellmatsM_H1_LUMPED[abs(ellmatsM_H1_LUMPED)<1e-14] = 0
        MAT['Mh'] = sparse(im_H1,jm_H1,ellmatsM_H1_LUMPED,INFO['sizeM'],INFO['sizeM'])
        MAT['Mh'].eliminate_zeros()
    return MAT




def h1b(MESH,BASIS,LISTS,Dict):
    
    space = Dict.get('space')
    size = Dict.get('size')
    edges = Dict.get('edges')
    
    if 'edges' in Dict.keys():
        edges = Dict.get('edges')
    else:
        edges = MESH.Boundary.Region
    
    indices = npy.argwhere(npy.in1d(MESH.Boundary.Region,edges))[:,0]
    
    p = MESH.p;    
    e = MESH.e[indices,:]; ne = e.shape[0]
    
    INFO = get_info(MESH,space)
    qp_B = INFO['qp_we_B'][0]; we_B = INFO['qp_we_B'][1]
    phi_H1_B = BASIS[space]['phi']; lphi_H1_B = len(phi_H1_B)
    
    H1_BOUNDARY_LIST_DOF = LISTS[space]['LIST_DOF'][indices,:]
    ellmatsB_H1 = npy.zeros((ne,lphi_H1_B*lphi_H1_B))
    phii_H1_B = npy.zeros((ne,lphi_H1_B))
    
    #####################################################################################
    # Mappings
    #####################################################################################
        
    e0 = e[:,0]; e1 = e[:,1]
    A0 =  p[e1,0]-p[e0,0]; A1 =  p[e1,1]-p[e0,1]
    detA = npy.sqrt(A0**2+A1**2)
    
    #####################################################################################
    # Mass matrix (over the edge)
    #####################################################################################
    
    for i in range(len(we_B)):
        for j in range(lphi_H1_B):
            phii_H1_B[:,j] = phi_H1_B[j](qp_B[i])

        ellmatsB_H1 = ellmatsB_H1 + we_B[i]*(assem_ellmats(phii_H1_B,phii_H1_B))*npy.abs(detA)[:,None]

    # ellmatsB_H1 = ellmatsB_H1[indices,:]
    
    ib_H1,jb_H1 = create_indices(H1_BOUNDARY_LIST_DOF,H1_BOUNDARY_LIST_DOF)
    
    ellmatsB_H1[abs(ellmatsB_H1)<1e-14] = 0
    B = sparse(ib_H1,jb_H1,ellmatsB_H1,size,size)
    B.eliminate_zeros()
    
    return B


def assem_ellmats(basis1,basis2): # for fast assembly
    
    basis1 = npy.r_[basis1] # make sure its of type numpy.array
    basis2 = npy.r_[basis2] # make sure its of type numpy.array
    
    if len(basis1.shape) == 1: basis1 = basis1[:,None]
    if len(basis2.shape) == 1: basis2 = basis2[:,None]

    lb1 = basis1.shape[1]
    lb2 = basis2.shape[1]
    nelem = basis1.shape[0]

    AAx = npy.tile(basis1,(1,lb2))
    AAy = npy.tile(basis2,(lb1,1)).reshape(nelem,lb1*lb2,order='F')
    
    return AAx*AAy

def create_indices(ind1,ind2):
    if ind1.ndim == 1: ind1 = ind1[:,None]
    if ind2.ndim == 1: ind2 = ind2[:,None]

    ii = npy.tile(ind2,(ind1.shape[1],1))
    jj = npy.tile(ind1,(1,ind2.shape[1]))

    return ii,jj

def sparse(i, j, v, m, n):
    """
    Create and compressing a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values 
            Size n1
        j: 1-D array representing the index 2 values 
            Size n1
        v: 1-D array representing the values 
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return sp.csc_matrix((v.flatten(order='F'), (i.flatten(order='F'), j.flatten(order='F'))), shape=(m, n))