
from .. import quadrature
import numpy as npy
from scipy import sparse as sp
# import matplotlib.pyplot as plt
import time

def get_info(MESH,space):
    
    def get_info_trig(space):
        
        INFO = {}
        INFO['TRIG'] = {}
            
        if space == 'P1':
            INFO['TRIG']['space'] = 'P1'
            INFO['TRIG']['space_dx'] = 'P0'
            INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
            INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
            INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
            
        if space == 'P2':
            INFO['TRIG']['space'] = 'P2'
            INFO['TRIG']['space_dx'] = 'P1'
            INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 4)
            # INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
            INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 2)
            
        return INFO
    
    
    INFO = {}
    
    ###########################################################################
    if space == 'P1': 
        INFO = INFO | get_info_trig('P1')
        INFO['sizeM'] = MESH.np
        INFO['sizeD'] = MESH.nt
        INFO['qp_we_B'] = quadrature.one_d(order = 2)
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P2': 
        INFO = INFO | get_info_trig('P2')
        INFO['sizeM'] = MESH.np + MESH.NoEdges
        INFO['sizeD'] = 3*MESH.nt
        INFO['qp_we_B'] = quadrature.one_d(order = 5) # 4 would suffice
    ###########################################################################
    
    return INFO




def h1(MESH,BASIS,LISTS,Dict):
    space = Dict.get('space')
    time_elapsed = time.time()
    
    INFO = get_info(MESH,space)
    
    MAT = h1_trig(MESH,BASIS,LISTS,INFO,Dict)
    
    elapsed = time.time()-time_elapsed
    print('Assembling ' + space + ' took ' + str(elapsed)[0:5] + ' seconds.')
    return MAT

def h1_trig(MESH,BASIS,LISTS,INFO,Dict):
    
    matrix = Dict.get('matrix')
    if 'coeff' in Dict.keys():
        coeff = Dict.get('coeff')
    else:
        coeff = lambda x,y : 1+x*0+y*0
        
    if 'coeff_const' in Dict.keys():
        coeff_const = Dict.get('coeff_const')
    else:
        coeff_const = npy.ones(shape = MESH.nt)
        
    if 'regions' in Dict.keys():
        regions = Dict.get('regions')
    else:
        regions = MESH.RegionsT
    
    indices = npy.argwhere(npy.in1d(MESH.RegionsT,regions))[:,0]

    p = MESH.p;
    t = MESH.t[indices,:]; nt = t.shape[0]
    
    
    spaceTrig = INFO['TRIG']['space'];
    spaceTrig_dx = INFO['TRIG']['space_dx'];
    sizeM = INFO['sizeM']
    sizeD = INFO['sizeD']
    qp_M = INFO['TRIG']['qp_we_M'][0]; we_M = INFO['TRIG']['qp_we_M'][1]
    qp_K = INFO['TRIG']['qp_we_K'][0]; we_K = INFO['TRIG']['qp_we_K'][1]

    phi_H1 = BASIS[spaceTrig]['TRIG']['phi']; lphi_H1 = len(phi_H1)
    dphi_H1 = BASIS[spaceTrig]['TRIG']['dphi']; ldphi_H1 = len(dphi_H1)
    phi_L2 = BASIS[spaceTrig_dx]['TRIG']['phi']; lphi_L2 = len(phi_L2)

    H1_LIST_DOF = LISTS[spaceTrig]['TRIG']['LIST_DOF'][indices,:]
    H1_DX_LIST_DOF = LISTS[spaceTrig_dx]['TRIG']['LIST_DOF'][indices,:]

    phiix_H1 = npy.zeros((nt,lphi_H1))
    phiiy_H1 = npy.zeros((nt,lphi_H1))
    phii_H1 = npy.zeros((nt,lphi_H1))
    phii_L2 = npy.zeros((nt,lphi_L2))

    #####################################################################################
    # Mappings
    #####################################################################################

    t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]
    A00 = p[t1,0]-p[t0,0]; A01 = p[t2,0]-p[t0,0]
    A10 = p[t1,1]-p[t0,1]; A11 = p[t2,1]-p[t0,1]
    detA = A00*A11-A01*A10
    
    
    im_H1,jm_H1 = create_indices(H1_LIST_DOF,H1_LIST_DOF)
    
    #####################################################################################
    # Mass matrix
    #####################################################################################
    
    if matrix == 'M':
        nqp = len(we_M)
        ellmatsM_H1 = npy.zeros((nqp*nt,lphi_H1))
        # phii_H1 = npy.zeros((nt,lphi_H1))
        
        qp_list_DOF = npy.r_[0:nt*nqp].reshape(nt,nqp)
        
        for j in range(lphi_H1):
            for i in range(nqp):
                ellmatsM_H1[i*nt:(i+1)*nt,j] = phi_H1[j](qp_M[0,i],qp_M[1,i])*npy.sqrt(1/2*npy.abs(detA)*we_M[i])            
        
        im_H1,jm_H1 = create_indices(H1_LIST_DOF,qp_list_DOF)
        M = sparse(im_H1,jm_H1,ellmatsM_H1,sizeM,nqp*nt)
        M.eliminate_zeros()
        return M

    #####################################################################################
    # Stiffness matrix
    #####################################################################################

    if matrix == 'K':
        ellmatsKxx_H1 = npy.zeros((nt,ldphi_H1*ldphi_H1))
        ellmatsKyy_H1 = npy.zeros((nt,ldphi_H1*ldphi_H1))
        ellmatsKxy_H1 = npy.zeros((nt,ldphi_H1*ldphi_H1))
        ellmatsKyx_H1 = npy.zeros((nt,ldphi_H1*ldphi_H1))
        
        for i in range(len(we_K)):
            qpT_i_1 = A00*qp_K[0,i]+A01*qp_K[1,i]+p[t0,0]
            qpT_i_2 = A10*qp_K[0,i]+A11*qp_K[1,i]+p[t0,1]
            coeff_qpT_i = coeff(qpT_i_1,qpT_i_2)
            
            for j in range(lphi_H1):
                dphii_H1 = dphi_H1[j](qp_K[0,i],qp_K[1,i])
                phiix_H1[:,j] = 1/detA*( A11*dphii_H1[0] -A10*dphii_H1[1])
                phiiy_H1[:,j] = 1/detA*(-A01*dphii_H1[0] +A00*dphii_H1[1])
    
            ellmatsKxx_H1 = ellmatsKxx_H1 + 1/2*we_K[i]*assem_ellmats(phiix_H1,phiix_H1)*npy.abs(detA)[:,None]*coeff_qpT_i[:,None]*coeff_const[indices,None]
            ellmatsKyy_H1 = ellmatsKyy_H1 + 1/2*we_K[i]*assem_ellmats(phiiy_H1,phiiy_H1)*npy.abs(detA)[:,None]*coeff_qpT_i[:,None]*coeff_const[indices,None]
            ellmatsKxy_H1 = ellmatsKxy_H1 + 1/2*we_K[i]*assem_ellmats(phiix_H1,phiiy_H1)*npy.abs(detA)[:,None]*coeff_qpT_i[:,None]*coeff_const[indices,None]
            ellmatsKyx_H1 = ellmatsKyx_H1 + 1/2*we_K[i]*assem_ellmats(phiiy_H1,phiix_H1)*npy.abs(detA)[:,None]*coeff_qpT_i[:,None]*coeff_const[indices,None]
            
        ellmatsKxx_H1[abs(ellmatsKxx_H1)<1e-14] = 0
        ellmatsKyy_H1[abs(ellmatsKyy_H1)<1e-14] = 0
        ellmatsKxy_H1[abs(ellmatsKxy_H1)<1e-14] = 0
        ellmatsKyx_H1[abs(ellmatsKyx_H1)<1e-14] = 0
        
        Kxx = sparse(im_H1,jm_H1,ellmatsKxx_H1,sizeM,sizeM); Kxx.eliminate_zeros()
        Kyy = sparse(im_H1,jm_H1,ellmatsKyy_H1,sizeM,sizeM); Kyy.eliminate_zeros()
        Kxy = sparse(im_H1,jm_H1,ellmatsKxy_H1,sizeM,sizeM); Kxy.eliminate_zeros()
        Kyx = sparse(im_H1,jm_H1,ellmatsKyx_H1,sizeM,sizeM); Kyx.eliminate_zeros()
        
        return Kxx, Kyy, Kxy, Kyx

    #####################################################################################
    # grad trafo matrix
    #####################################################################################

    if matrix == 'C':
        ellmatsCx_H1 = npy.zeros((nt,lphi_H1*lphi_L2))
        ellmatsCy_H1 = npy.zeros((nt,lphi_H1*lphi_L2))
        
        for i in range(len(we_M)):
            qpT_i_1 = A00*qp_M[0,i]+A01*qp_M[1,i]+p[t0,0]
            qpT_i_2 = A10*qp_M[0,i]+A11*qp_M[1,i]+p[t0,1]
            coeff_qpT_i = coeff(qpT_i_1,qpT_i_2)
            
            for j in range(lphi_H1):
                dphii_H1 = dphi_H1[j](qp_M[0,i],qp_M[1,i])
                phiix_H1[:,j] = 1/detA*( A11*dphii_H1[0] -A10*dphii_H1[1])
                phiiy_H1[:,j] = 1/detA*(-A01*dphii_H1[0] +A00*dphii_H1[1])
            for j in range(lphi_L2):
                phii_L2[:,j] = phi_L2[j](qp_M[0,i],qp_M[1,i])
    
            ellmatsCx_H1 = ellmatsCx_H1 + 1/2*we_M[i]*assem_ellmats(phiix_H1,phii_L2)*npy.abs(detA)[:,None]*coeff_qpT_i[:,None]*coeff_const[indices,None]
            ellmatsCy_H1 = ellmatsCy_H1 + 1/2*we_M[i]*assem_ellmats(phiiy_H1,phii_L2)*npy.abs(detA)[:,None]*coeff_qpT_i[:,None]*coeff_const[indices,None]
            
        ellmatsCx_H1[abs(ellmatsCx_H1)<1e-14] = 0
        ellmatsCy_H1[abs(ellmatsCy_H1)<1e-14] = 0
        
        ic_H1,jc_H1 = create_indices(H1_LIST_DOF,H1_DX_LIST_DOF)
        Cx = sparse(ic_H1,jc_H1,ellmatsCx_H1,sizeD,sizeM); Cx.eliminate_zeros()
        Cy = sparse(ic_H1,jc_H1,ellmatsCy_H1,sizeD,sizeM); Cy.eliminate_zeros()
        return Cx,Cy
        
    #####################################################################################
    # Lumped mass matrix
    #####################################################################################

    if matrix == 'Mh':
        ellmatsM_H1_LUMPED = npy.zeros((nt,lphi_H1*lphi_H1))
        
        if 'qp_we_Mh' in INFO['TRIG']:
            qp_Mh = INFO['TRIG']['qp_we_Mh'][0]; we_Mh = INFO['TRIG']['qp_we_Mh'][1]
            for i in range(len(we_Mh)):
                qpT_i_1 = A00*qp_K[0,i]+A01*qp_K[1,i]+p[t0,0]
                qpT_i_2 = A10*qp_K[0,i]+A11*qp_K[1,i]+p[t0,1]
                coeff_qpT_i = coeff(qpT_i_1,qpT_i_2)
                for j in range(lphi_H1):
                    phii_H1[:,j] = phi_H1[j](qp_Mh[0,i],qp_Mh[1,i])
    
                ellmatsM_H1_LUMPED = ellmatsM_H1_LUMPED + 1/2*we_M[i]*(assem_ellmats(phii_H1,phii_H1))*npy.abs(detA)[:,None]*coeff_qpT_i[:,None]*coeff_const[indices,None]
            
            ellmatsM_H1_LUMPED[abs(ellmatsM_H1_LUMPED)<1e-14] = 0
            Mh = sparse(im_H1,jm_H1,ellmatsM_H1_LUMPED,sizeM,sizeM)
            Mh.eliminate_zeros()
            return Mh
        





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
    phi_H1_B = BASIS[space]['B']['phi']; lphi_H1_B = len(phi_H1_B)
    
    H1_BOUNDARY_LIST_DOF = LISTS[space]['B']['LIST_DOF'][indices,:]
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

    ii = npy.tile(ind1,(1,ind2.shape[1]))
    jj = npy.tile(ind2,(ind1.shape[1],1))

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