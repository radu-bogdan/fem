
from re import I
from . import quadrature
import numpy as npy
from scipy import sparse as sp
# import matplotlib.pyplot as plt
import time

def get_info(MESH,space):
    
    def get_info_trig(space):
        
        INFO = {}
        INFO['TRIG'] = {}
        if space == 'RT0':
            INFO['TRIG']['space'] = 'RT0'
            INFO['TRIG']['mixedSpace'] = 'P0'
            INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
            INFO['TRIG']['qp_we_C'] = quadrature.dunavant(order = 1)
            INFO['TRIG']['qp_we_D'] = quadrature.dunavant(order = 0)
            INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
            INFO['TRIG']['qp_we_MT'] = quadrature.dunavant(order = 2)

        if space == 'BDM1':
            INFO['TRIG']['space'] = 'BDM1'
            INFO['TRIG']['mixedSpace'] = 'P0'
            INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
            # INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = '1kek')
            INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
            INFO['TRIG']['qp_we_C'] = quadrature.dunavant(order = 1)
            INFO['TRIG']['qp_we_D'] = quadrature.dunavant(order = 0)
            INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
            INFO['TRIG']['qp_we_MT'] = quadrature.dunavant(order = 2)
            
        if space == 'EJ1':
            INFO['TRIG']['space'] = 'EJ1'
            INFO['TRIG']['mixedSpace'] = 'P1d'
            INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 4)
            INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = '2m')
            INFO['TRIG']['qp_we_C'] = quadrature.dunavant(order = 2)
            INFO['TRIG']['qp_we_D'] = quadrature.dunavant(order = 2)
            INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 2)
            INFO['TRIG']['qp_we_MT'] = quadrature.dunavant(order = 4)
            
        if space == 'RT1':
            INFO['TRIG']['space'] = 'RT1'
            INFO['TRIG']['mixedSpace'] = 'P1d'
            INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 4)
            INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = '2l')
            INFO['TRIG']['qp_we_C'] = quadrature.dunavant(order = 1)
            INFO['TRIG']['qp_we_D'] = quadrature.dunavant(order = 2)
            INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
            INFO['TRIG']['qp_we_MT'] = quadrature.dunavant(order = 2)
            
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
            
        if space == 'P1d':
            INFO['TRIG']['space'] = 'P1d'
            INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
            INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
            INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
            
        return INFO

    
    
    def get_info_quad(space):
        INFO = {}
        INFO['QUAD'] = {}
        if space == 'RT0':
            INFO['QUAD']['space'] = 'RT0'
            INFO['QUAD']['mixedSpace'] = 'Q0'
            INFO['QUAD']['qp_we_M'] = quadrature.quadrule(order = 3) # 2 would be enough, but quads are weird.
            INFO['QUAD']['qp_we_C'] = quadrature.quadrule(order = 1)
            INFO['QUAD']['qp_we_D'] = quadrature.quadrule(order = 0)
            INFO['QUAD']['qp_we_K'] = quadrature.quadrule(order = 0)
            INFO['QUAD']['qp_we_MT'] = quadrature.quadrule(order = 3) # 2 would be enough, but quads are weird.

        if space == 'BDM1':
            INFO['QUAD']['space'] = 'BDM1'
            INFO['QUAD']['mixedSpace'] = 'Q0'
            INFO['QUAD']['qp_we_M'] = quadrature.quadrule(order = 3) # 2 would be enough, but quads are weird.
            INFO['QUAD']['qp_we_Mh'] = quadrature.quadrule(order = 1)
            INFO['QUAD']['qp_we_C'] = quadrature.quadrule(order = 1)
            INFO['QUAD']['qp_we_D'] = quadrature.quadrule(order = 0)
            INFO['QUAD']['qp_we_K'] = quadrature.quadrule(order = 0)
            INFO['QUAD']['qp_we_MT'] = quadrature.quadrule(order = 3) # 2 would be enough, but quads are weird.
            
        if space == 'BDFM1':
            INFO['QUAD']['space'] = 'BDFM1'
            INFO['QUAD']['mixedSpace'] = 'P1d'
            INFO['QUAD']['qp_we_M'] = quadrature.quadrule(order = 5) # 4 reicht auch
            INFO['QUAD']['qp_we_Mh'] = quadrature.quadrule(order = '3l')
            INFO['QUAD']['qp_we_C'] = quadrature.quadrule(order = 3) # 2 reicht auch
            INFO['QUAD']['qp_we_D'] = quadrature.quadrule(order = 3)
            INFO['QUAD']['qp_we_K'] = quadrature.quadrule(order = 3) # 2 reicht auch
            INFO['QUAD']['qp_we_MT'] = quadrature.quadrule(order = 3)
        
        if space == 'Q1':
            INFO['QUAD']['space'] = 'Q1'
            INFO['QUAD']['qp_we_M'] = quadrature.quadrule(order = 3)
            INFO['QUAD']['qp_we_Mh'] = quadrature.quadrule(order = 1)
            INFO['QUAD']['qp_we_K'] = quadrature.quadrule(order = 3)
        
        if space == 'Q1d':
            INFO['QUAD']['space'] = 'Q1d'
            INFO['QUAD']['qp_we_M'] = quadrature.quadrule(order = 3)
            INFO['QUAD']['qp_we_Mh'] = quadrature.quadrule(order = 1)
            INFO['QUAD']['qp_we_K'] = quadrature.quadrule(order = 3)
            
        return INFO
    
    INFO = {}
    
    ###########################################################################
    if space == 'RT0-RT0':
        INFO = INFO | get_info_trig('RT0')
        INFO = INFO | get_info_quad('RT0')
        INFO['sizeM'] = MESH.NoEdges
        INFO['sizeD_trig'] = MESH.nt
        INFO['sizeD_quad'] = MESH.nq
    ###########################################################################
    
    
    ###########################################################################
    if space == 'BDM1-BDM1':
        INFO = INFO | get_info_trig('BDM1')
        INFO = INFO | get_info_quad('BDM1')
        INFO['sizeM'] = 2*MESH.NoEdges
        INFO['sizeD_trig'] = MESH.nt
        INFO['sizeD_quad'] = MESH.nq
    ###########################################################################    
    
    
    ###########################################################################
    if space == 'RT1-BDFM1':
        INFO = INFO | get_info_trig('RT1')
        INFO = INFO | get_info_quad('BDFM1')
        INFO['sizeM'] = 2*MESH.NoEdges + 2*MESH.nt + 2*MESH.nq
        INFO['sizeD_trig'] = 3*MESH.nt
        INFO['sizeD_quad'] = 3*MESH.nq
    ###########################################################################
    
    
    ###########################################################################
    if space == 'EJ1-RT0': 
        INFO = INFO | get_info_trig('EJ1')
        INFO = INFO | get_info_quad('RT0')
        INFO['sizeM'] = MESH.NoEdges + 3*MESH.nt
        INFO['sizeD_trig'] = 3*MESH.nt
        INFO['sizeD_quad'] = MESH.nq
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P1-Q1': 
        INFO = INFO | get_info_trig('P1')
        INFO = INFO | get_info_quad('Q1')
        INFO['sizeM'] = MESH.np
        INFO['sizeD'] = MESH.nt # +MESH.nq
    ###########################################################################
    
    
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
    
    
    ###########################################################################
    if space == 'P1d-Q1d': 
        INFO = INFO | get_info_trig('P1d')
        INFO = INFO | get_info_quad('Q1d')
        INFO['sizeM'] = 3*MESH.nt+4*MESH.nq
        INFO['list_shift'] = 3*MESH.nt
    ###########################################################################
    
    if space == 'P1b':
        INFO['qp_we_B'] = quadrature.one_d(order = 2)
    
    return INFO





def hdiv(MESH,BASIS,LISTS,space):
    time_elapsed = time.time()
    MAT = {}; MAT[space] = {}
    MAT_TRIG = {}; MAT_QUAD = {}
    
    t = MESH.t; q = MESH.q
    
    INFO = get_info(MESH,space)

    if t.shape[0] != 0: MAT_TRIG = hdiv_trig(MESH,BASIS,LISTS,INFO)
    if q.shape[0] != 0: MAT_QUAD = hdiv_quad(MESH,BASIS,LISTS,INFO)

    if t.shape[0] == 0: MAT[space] = MAT_QUAD
    if q.shape[0] == 0: MAT[space] = MAT_TRIG

    if t.shape[0] != 0 and q.shape[0] != 0:
        MAT_TRIG_Set = set(MAT_TRIG)
        MAT_QUAD_Set = set(MAT_QUAD)
        
        for field in MAT_TRIG_Set.intersection(MAT_QUAD_Set):
            MAT[space][field] = MAT_TRIG[field] + MAT_QUAD[field]
        
        if 'Mh' in MAT_TRIG_Set:
            MAT[space]['Mh_only_trig'] = MAT_TRIG['Mh'] + MAT_QUAD['M']
            
        if 'Mh' in MAT_QUAD_Set:
            MAT[space]['Mh_only_quad'] = MAT_TRIG['M'] + MAT_QUAD['Mh']
            
    elapsed = time.time()-time_elapsed
    print('Assembling ' + space + ' took ' + str(elapsed)[0:5] + ' seconds.')
    return MAT
    

    


def hdiv_trig(MESH,BASIS,LISTS,INFO):

    p = MESH.p; np = MESH.np
    t = MESH.t; nt = MESH.nt
    
    spaceTrig = INFO['TRIG']['space']; mixedSpace = INFO['TRIG']['mixedSpace']
    qp_M = INFO['TRIG']['qp_we_M'][0]; we_M = INFO['TRIG']['qp_we_M'][1]
    qp_C = INFO['TRIG']['qp_we_C'][0]; we_C = INFO['TRIG']['qp_we_C'][1]
    qp_K = INFO['TRIG']['qp_we_K'][0]; we_K = INFO['TRIG']['qp_we_K'][1]
    qp_D = INFO['TRIG']['qp_we_D'][0]; we_D = INFO['TRIG']['qp_we_D'][1]
    qp_MT = INFO['TRIG']['qp_we_MT'][0]; we_MT = INFO['TRIG']['qp_we_MT'][1]

    phi_HDIV = BASIS[spaceTrig]['TRIG']['phi']; lphi_HDIV = len(phi_HDIV)
    divphi_HDIV = BASIS[spaceTrig]['TRIG']['divphi']; ldivphi_HDIV = len(divphi_HDIV)
    phi_L2 = BASIS[mixedSpace]['TRIG']['phi']; lphi_L2 = len(phi_L2)

    HDIV_LIST_DOF = LISTS[spaceTrig]['TRIG']['LIST_DOF']
    DIRECTION_DOF = LISTS[spaceTrig]['TRIG']['DIRECTION_DOF']

    L2_LIST_DOF = LISTS[mixedSpace]['TRIG']['LIST_DOF']
    P1d_LIST_DOF = LISTS['P1d']['TRIG']['LIST_DOF']
    P1_LIST_DOF = LISTS['P1']['TRIG']['LIST_DOF']
    P0_LIST_DOF = LISTS['P0']['TRIG']['LIST_DOF']

    phi_P0 = BASIS['P0']['TRIG']['phi']; lphi_P0 = len(phi_P0)
    phi_P1 = BASIS['P1']['TRIG']['phi']; lphi_P1 = len(phi_P1)

    ellmatsM_HDIV = npy.zeros((nt,lphi_HDIV*lphi_HDIV))
    ellmatsM_HDIV_LUMPED = npy.zeros((nt,lphi_HDIV*lphi_HDIV))
    ellmatsK_HDIV = npy.zeros((nt,lphi_HDIV*lphi_HDIV))
    ellmatsM_HDIVx_P1d = npy.zeros((nt,lphi_P1*lphi_HDIV))
    ellmatsM_HDIVy_P1d = npy.zeros((nt,lphi_P1*lphi_HDIV))
    ellmatsC_HDIV_L2 = npy.zeros((nt,lphi_L2*lphi_HDIV))
    ellmatsD_L2 = npy.zeros((nt,lphi_L2*lphi_L2))

    phiix_HDIV = npy.zeros((nt,lphi_HDIV))
    phiiy_HDIV = npy.zeros((nt,lphi_HDIV))
    divphii_HDIV = npy.zeros((nt,lphi_HDIV))
    phii_L2 = npy.zeros((nt,lphi_L2))
    phii_P1 = npy.zeros((nt,lphi_P1))
    phii_P0 = npy.zeros((nt,lphi_P0))

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
        for j in range(lphi_HDIV):
            phii_HDIV = phi_HDIV[j](qp_M[0,i],qp_M[1,i])
            phiix_HDIV[:,j] = 1/detA*(A00*phii_HDIV[0] + A01*phii_HDIV[1])*DIRECTION_DOF[:,j]
            phiiy_HDIV[:,j] = 1/detA*(A10*phii_HDIV[0] + A11*phii_HDIV[1])*DIRECTION_DOF[:,j]

        ellmatsM_HDIV = ellmatsM_HDIV + 1/2*we_M[i]*(assem_ellmats(phiix_HDIV,phiix_HDIV) + assem_ellmats(phiiy_HDIV,phiiy_HDIV))*npy.abs(detA)[:,None]

    im_HDIV,jm_HDIV = create_indices(HDIV_LIST_DOF,HDIV_LIST_DOF)
    ellmatsM_HDIV[abs(ellmatsM_HDIV)<1e-14] = 0
    MAT['M'] = sparse(im_HDIV,jm_HDIV,ellmatsM_HDIV,INFO['sizeM'],INFO['sizeM'])
    MAT['M'].eliminate_zeros()

    #####################################################################################
    # Stiffness matrix
    #####################################################################################

    for i in range(len(we_K)):
        for j in range(lphi_HDIV):
            divphii_HDIV[:,j] = 1/detA*divphi_HDIV[j](qp_K[0,i],qp_K[1,i])*DIRECTION_DOF[:,j]

        ellmatsK_HDIV = ellmatsK_HDIV + 1/2*we_K[i]*assem_ellmats(divphii_HDIV,divphii_HDIV)*npy.abs(detA)[:,None]

    ellmatsK_HDIV[abs(ellmatsK_HDIV)<1e-14] = 0
    MAT['K'] = sparse(im_HDIV,jm_HDIV,ellmatsK_HDIV,INFO['sizeM'],INFO['sizeM'])
    MAT['K'].eliminate_zeros()

    #####################################################################################
    # Mixed matrix
    #####################################################################################

    for i in range(len(we_C)):
        for j in range(lphi_HDIV):
            phii_HDIV = phi_HDIV[j](qp_C[0,i],qp_C[1,i])
            phiix_HDIV[:,j] = 1/detA*(A00*phii_HDIV[0] + A01*phii_HDIV[1])*DIRECTION_DOF[:,j]
            phiiy_HDIV[:,j] = 1/detA*(A10*phii_HDIV[0] + A11*phii_HDIV[1])*DIRECTION_DOF[:,j]
        for j in range(lphi_L2):
            phii_L2[:,j] = phi_L2[j](qp_C[0,i],qp_C[1,i])

        ellmatsC_HDIV_L2 = ellmatsC_HDIV_L2 + 1/2*we_C[i]*assem_ellmats(divphii_HDIV,phii_L2)*npy.abs(detA)[:,None]
    
    ellmatsC_HDIV_L2[abs(ellmatsC_HDIV_L2)<1e-14] = 0
    ic_HDIV,jc_HDIV = create_indices(HDIV_LIST_DOF,L2_LIST_DOF)
    MAT['C'] = sparse(ic_HDIV,jc_HDIV,ellmatsC_HDIV_L2,INFO['sizeD_trig']+INFO['sizeD_quad'],INFO['sizeM'])
    MAT['C'].eliminate_zeros()
    
    #####################################################################################
    # Mass matrix mixed space
    #####################################################################################
    
    for i in range(len(we_D)):
        for j in range(lphi_L2):
            phii_L2[:,j] = phi_L2[j](qp_D[0,i],qp_D[1,i])

        ellmatsD_L2 = ellmatsD_L2 + 1/2*we_D[i]*assem_ellmats(phii_L2,phii_L2)*npy.abs(detA)[:,None]
    
    ellmatsD_L2[abs(ellmatsD_L2)<1e-14] = 0
    id_L2,jd_L2 = create_indices(L2_LIST_DOF,L2_LIST_DOF)
    MAT['D'] = sparse(id_L2,jd_L2,ellmatsD_L2,INFO['sizeD_trig']+INFO['sizeD_quad'],INFO['sizeD_trig']+INFO['sizeD_quad'])
    MAT['D'].eliminate_zeros()
    
    #####################################################################################
    # Basis transformation matrices
    #####################################################################################

    for i in range(len(we_MT)):
        for j in range(lphi_HDIV):
            phii_HDIV = phi_HDIV[j](qp_MT[0,i],qp_MT[1,i])
            phiix_HDIV[:,j] = 1/detA*(A00*phii_HDIV[0] + A01*phii_HDIV[1])*DIRECTION_DOF[:,j]
            phiiy_HDIV[:,j] = 1/detA*(A10*phii_HDIV[0] + A11*phii_HDIV[1])*DIRECTION_DOF[:,j]
        for j in range(lphi_P1):
            phii_P1[:,j] = phi_P1[j](qp_MT[0,i],qp_MT[1,i])

        ellmatsM_HDIVx_P1d = ellmatsM_HDIVx_P1d + 1/2*we_MT[i]*assem_ellmats(phiix_HDIV,phii_P1)*npy.abs(detA)[:,None]
        ellmatsM_HDIVy_P1d = ellmatsM_HDIVy_P1d + 1/2*we_MT[i]*assem_ellmats(phiiy_HDIV,phii_P1)*npy.abs(detA)[:,None]

    iMT_P1_HDIV,jMT_P1_HDIV = create_indices(HDIV_LIST_DOF,P1_LIST_DOF)
    ellmatsM_HDIVx_P1d[abs(ellmatsM_HDIVx_P1d)<1e-14] = 0
    ellmatsM_HDIVy_P1d[abs(ellmatsM_HDIVy_P1d)<1e-14] = 0
    
    MAT['Mx_P1'] = sparse(iMT_P1_HDIV,jMT_P1_HDIV,ellmatsM_HDIVx_P1d,MESH.np,INFO['sizeM'])
    MAT['My_P1'] = sparse(iMT_P1_HDIV,jMT_P1_HDIV,ellmatsM_HDIVy_P1d,MESH.np,INFO['sizeM'])

    iMT_P1d_HDIV,jMT_P1d_HDIV = create_indices(HDIV_LIST_DOF,P1d_LIST_DOF)
    MAT['Mx_P1d_Q1d'] = sparse(iMT_P1d_HDIV,jMT_P1d_HDIV,ellmatsM_HDIVx_P1d,3*MESH.nt+4*MESH.nq,INFO['sizeM'])
    MAT['My_P1d_Q1d'] = sparse(iMT_P1d_HDIV,jMT_P1d_HDIV,ellmatsM_HDIVy_P1d,3*MESH.nt+4*MESH.nq,INFO['sizeM'])

    MAT['Mx_P1'].eliminate_zeros(); MAT['My_P1'].eliminate_zeros()
    MAT['Mx_P1d_Q1d'].eliminate_zeros(); MAT['My_P1d_Q1d'].eliminate_zeros()

    #####################################################################################
    # Lumped mass matrix
    #####################################################################################

    # if 'qp_Mh' in locals():
    if 'qp_we_Mh' in INFO['TRIG']:
        qp_Mh = INFO['TRIG']['qp_we_Mh'][0]; we_Mh = INFO['TRIG']['qp_we_Mh'][1]
        for i in range(len(we_Mh)):
            for j in range(lphi_HDIV):
                phii_HDIV = phi_HDIV[j](qp_Mh[0,i],qp_Mh[1,i])
                phiix_HDIV[:,j] = 1/detA*(A00*phii_HDIV[0] + A01*phii_HDIV[1])*DIRECTION_DOF[:,j]
                phiiy_HDIV[:,j] = 1/detA*(A10*phii_HDIV[0] + A11*phii_HDIV[1])*DIRECTION_DOF[:,j]

            ellmatsM_HDIV_LUMPED = ellmatsM_HDIV_LUMPED + 1/2*we_Mh[i]*(assem_ellmats(phiix_HDIV,phiix_HDIV) + assem_ellmats(phiiy_HDIV,phiiy_HDIV))*npy.abs(detA)[:,None]
        
        ellmatsM_HDIV_LUMPED[abs(ellmatsM_HDIV_LUMPED)<1e-14] = 0
        MAT['Mh'] = sparse(im_HDIV,jm_HDIV,ellmatsM_HDIV_LUMPED,INFO['sizeM'],INFO['sizeM'])
        MAT['Mh'].eliminate_zeros()
    return MAT



def hdiv_quad(MESH,BASIS,LISTS,INFO):

    p = MESH.p; np = MESH.np
    q = MESH.q; nq = MESH.nq
    
    spaceQuad = INFO['QUAD']['space']; mixedSpace = INFO['QUAD']['mixedSpace']
    qp_M = INFO['QUAD']['qp_we_M'][0]; we_M = INFO['QUAD']['qp_we_M'][1]
    qp_C = INFO['QUAD']['qp_we_C'][0]; we_C = INFO['QUAD']['qp_we_C'][1]
    qp_K = INFO['QUAD']['qp_we_K'][0]; we_K = INFO['QUAD']['qp_we_K'][1]
    qp_D = INFO['QUAD']['qp_we_D'][0]; we_D = INFO['QUAD']['qp_we_D'][1]
    qp_MT = INFO['QUAD']['qp_we_MT'][0]; we_MT = INFO['QUAD']['qp_we_MT'][1]

    phi_HDIV = BASIS[spaceQuad]['QUAD']['phi']; lphi_HDIV = len(phi_HDIV)
    divphi_HDIV = BASIS[spaceQuad]['QUAD']['divphi']; ldivphi_HDIV = len(divphi_HDIV)
    phi_L2 = BASIS[mixedSpace]['QUAD']['phi']; lphi_L2 = len(phi_L2)

    HDIV_LIST_DOF = LISTS[spaceQuad]['QUAD']['LIST_DOF']
    DIRECTION_DOF = LISTS[spaceQuad]['QUAD']['DIRECTION_DOF']

    L2_LIST_DOF = LISTS[mixedSpace]['QUAD']['LIST_DOF']
    Q1d_LIST_DOF = LISTS['Q1d']['QUAD']['LIST_DOF']
    Q1_LIST_DOF = LISTS['Q1']['QUAD']['LIST_DOF']
    Q0_LIST_DOF = LISTS['Q0']['QUAD']['LIST_DOF']

    phi_Q0 = BASIS['Q0']['QUAD']['phi']; lphi_Q0 = len(phi_Q0)
    phi_Q1 = BASIS['Q1']['QUAD']['phi']; lphi_Q1 = len(phi_Q1)

    ellmatsM_HDIV = npy.zeros((nq,lphi_HDIV*lphi_HDIV))
    ellmatsM_HDIV_LUMPED = npy.zeros((nq,lphi_HDIV*lphi_HDIV))
    ellmatsK_HDIV = npy.zeros((nq,lphi_HDIV*lphi_HDIV))
    ellmatsM_HDIVx_P1d = npy.zeros((nq,lphi_Q1*lphi_HDIV))
    ellmatsM_HDIVy_P1d = npy.zeros((nq,lphi_Q1*lphi_HDIV))
    ellmatsC_HDIV_L2 = npy.zeros((nq,lphi_L2*lphi_HDIV))
    ellmatsD_L2 = npy.zeros((nq,lphi_L2*lphi_L2))

    phiix_HDIV = npy.zeros((nq,lphi_HDIV))
    phiiy_HDIV = npy.zeros((nq,lphi_HDIV))
    divphii_HDIV = npy.zeros((nq,lphi_HDIV))
    phii_L2 = npy.zeros((nq,lphi_L2))
    phii_Q1 = npy.zeros((nq,lphi_Q1))
    phii_Q0 = npy.zeros((nq,lphi_Q0))


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
        Q00i = Q00(qp_M[0,i],qp_M[1,i]); Q10i = Q10(qp_M[0,i],qp_M[1,i])
        Q01i = Q01(qp_M[0,i],qp_M[1,i]); Q11i = Q11(qp_M[0,i],qp_M[1,i])
        
        for j in range(lphi_HDIV):
            phii_HDIV = phi_HDIV[j](qp_M[0,i],qp_M[1,i])
            phiix_HDIV[:,j] = 1/detQi*(Q00i*phii_HDIV[0] + Q10i*phii_HDIV[1])*DIRECTION_DOF[:,j]
            phiiy_HDIV[:,j] = 1/detQi*(Q01i*phii_HDIV[0] + Q11i*phii_HDIV[1])*DIRECTION_DOF[:,j]

        ellmatsM_HDIV = ellmatsM_HDIV + we_M[i]*(assem_ellmats(phiix_HDIV,phiix_HDIV) + assem_ellmats(phiiy_HDIV,phiiy_HDIV))*npy.abs(detQi)[:,None]

    im_HDIV,jm_HDIV = create_indices(HDIV_LIST_DOF,HDIV_LIST_DOF)
    ellmatsM_HDIV[abs(ellmatsM_HDIV)<1e-14] = 0
    MAT['M'] = sparse(im_HDIV,jm_HDIV,ellmatsM_HDIV,INFO['sizeM'],INFO['sizeM'])
    MAT['M'].eliminate_zeros()

    #####################################################################################
    # Stiffness matrix
    #####################################################################################

    for i in range(len(we_K)):
        detQi = detQ(qp_K[0,i],qp_K[1,i])

        for j in range(lphi_HDIV):
            divphii_HDIV[:,j] = 1/detQi*divphi_HDIV[j](qp_K[0,i],qp_K[1,i])*DIRECTION_DOF[:,j]

        ellmatsK_HDIV = ellmatsK_HDIV + we_K[i]*assem_ellmats(divphii_HDIV,divphii_HDIV)*npy.abs(detQi)[:,None]

    ellmatsK_HDIV[abs(ellmatsK_HDIV)<1e-14] = 0
    MAT['K'] = sparse(im_HDIV,jm_HDIV,ellmatsK_HDIV,INFO['sizeM'],INFO['sizeM'])
    MAT['K'].eliminate_zeros()

    #####################################################################################
    # Mixed matrix
    #####################################################################################

    for i in range(len(we_C)):
        detQi = detQ(qp_C[0,i],qp_C[1,i])
        Q00i = Q00(qp_C[0,i],qp_C[1,i]); Q10i = Q10(qp_C[0,i],qp_C[1,i])
        Q01i = Q01(qp_C[0,i],qp_C[1,i]); Q11i = Q11(qp_C[0,i],qp_C[1,i])

        for j in range(lphi_HDIV):
            phii_HDIV = phi_HDIV[j](qp_C[0,i],qp_C[1,i])
            phiix_HDIV[:,j] = 1/detQi*(Q00i*phii_HDIV[0] + Q01i*phii_HDIV[1])*DIRECTION_DOF[:,j]
            phiiy_HDIV[:,j] = 1/detQi*(Q10i*phii_HDIV[0] + Q11i*phii_HDIV[1])*DIRECTION_DOF[:,j]
        for j in range(lphi_L2):
            phii_L2[:,j] = phi_L2[j](qp_C[0,i],qp_C[1,i])

        ellmatsC_HDIV_L2 = ellmatsC_HDIV_L2 + we_C[i]*assem_ellmats(divphii_HDIV,phii_L2)*npy.abs(detQi)[:,None]

    ic_HDIV,jc_HDIV = create_indices(HDIV_LIST_DOF,L2_LIST_DOF)
    ellmatsC_HDIV_L2[abs(ellmatsC_HDIV_L2)<1e-14] = 0
    MAT['C'] = sparse(INFO['sizeD_trig']+ic_HDIV,jc_HDIV,ellmatsC_HDIV_L2,INFO['sizeD_trig']+INFO['sizeD_quad'],INFO['sizeM']) # vll doch auf ic_HDIV
    MAT['C'].eliminate_zeros()
    
    #####################################################################################
    # Mass matrix mixed space
    #####################################################################################
    
    for i in range(len(we_D)):
        detQi = detQ(qp_D[0,i],qp_D[1,i])
        Q00i = Q00(qp_D[0,i],qp_D[1,i]); Q10i = Q10(qp_D[0,i],qp_D[1,i])
        Q01i = Q01(qp_D[0,i],qp_D[1,i]); Q11i = Q11(qp_D[0,i],qp_D[1,i])
        
        for j in range(lphi_L2):
            phii_L2[:,j] = phi_L2[j](qp_D[0,i],qp_D[1,i])

        ellmatsD_L2 = ellmatsD_L2 + we_D[i]*assem_ellmats(phii_L2,phii_L2)*npy.abs(detQi)[:,None]
    
    ellmatsD_L2[abs(ellmatsD_L2)<1e-14] = 0
    id_L2,jd_L2 = create_indices(L2_LIST_DOF,L2_LIST_DOF)
    MAT['D'] = sparse(id_L2,jd_L2,ellmatsD_L2,INFO['sizeD_trig']+INFO['sizeD_quad'],INFO['sizeD_trig']+INFO['sizeD_quad'])
    MAT['D'].eliminate_zeros()

    #####################################################################################
    # Basis transformation matrices
    #####################################################################################

    for i in range(len(we_MT)):
        detQi = detQ(qp_C[0,i],qp_C[1,i])
        Q00i = Q00(qp_C[0,i],qp_C[1,i]); Q10i = Q10(qp_C[0,i],qp_C[1,i])
        Q01i = Q01(qp_C[0,i],qp_C[1,i]); Q11i = Q11(qp_C[0,i],qp_C[1,i])

        for j in range(lphi_HDIV):
            phii_HDIV = phi_HDIV[j](qp_MT[0,i],qp_MT[1,i])
            phiix_HDIV[:,j] = 1/detQi*(Q00i*phii_HDIV[0] + Q01i*phii_HDIV[1])*DIRECTION_DOF[:,j]
            phiiy_HDIV[:,j] = 1/detQi*(Q10i*phii_HDIV[0] + Q11i*phii_HDIV[1])*DIRECTION_DOF[:,j]
        for j in range(lphi_Q1):
            phii_Q1[:,j] = phi_Q1[j](qp_MT[0,i],qp_MT[1,i])

        ellmatsM_HDIVx_P1d = ellmatsM_HDIVx_P1d + we_MT[i]*assem_ellmats(phiix_HDIV,phii_Q1)*npy.abs(detQi)[:,None]
        ellmatsM_HDIVy_P1d = ellmatsM_HDIVy_P1d + we_MT[i]*assem_ellmats(phiiy_HDIV,phii_Q1)*npy.abs(detQi)[:,None]

    ellmatsM_HDIVx_P1d[abs(ellmatsM_HDIVx_P1d)<1e-14] = 0
    ellmatsM_HDIVy_P1d[abs(ellmatsM_HDIVy_P1d)<1e-14] = 0

    iMT_P1_HDIV,jMT_P1_HDIV = create_indices(HDIV_LIST_DOF,Q1_LIST_DOF)
    MAT['Mx_P1'] = sparse(iMT_P1_HDIV,jMT_P1_HDIV,ellmatsM_HDIVx_P1d,MESH.np,INFO['sizeM']) # hier müssen die indices noch verschoben werden !
    MAT['My_P1'] = sparse(iMT_P1_HDIV,jMT_P1_HDIV,ellmatsM_HDIVy_P1d,MESH.np,INFO['sizeM']) # hier müssen die indices noch verschoben werden !

    iMT_P1d_HDIV,jMT_P1d_HDIV = create_indices(HDIV_LIST_DOF,3*MESH.nt+Q1d_LIST_DOF)
    MAT['Mx_P1d_Q1d'] = sparse(iMT_P1d_HDIV,jMT_P1d_HDIV,ellmatsM_HDIVx_P1d,3*MESH.nt+4*MESH.nq,INFO['sizeM']) # hier müssen die indices noch verschoben werden !
    MAT['My_P1d_Q1d'] = sparse(iMT_P1d_HDIV,jMT_P1d_HDIV,ellmatsM_HDIVy_P1d,3*MESH.nt+4*MESH.nq,INFO['sizeM']) # hier müssen die indices noch verschoben werden !

    MAT['Mx_P1'].eliminate_zeros(); MAT['My_P1'].eliminate_zeros()
    MAT['Mx_P1d_Q1d'].eliminate_zeros(); MAT['My_P1d_Q1d'].eliminate_zeros()

    #####################################################################################
    # Lumped mass matrix
    #####################################################################################

    # if 'qp_Mh' in locals():
    if 'qp_we_Mh' in INFO['QUAD']:
        qp_Mh = INFO['QUAD']['qp_we_Mh'][0]; we_Mh = INFO['QUAD']['qp_we_Mh'][1]
        for i in range(len(we_Mh)):
            detQi = detQ(qp_Mh[0,i],qp_Mh[1,i])
            
            Q00i = Q00(qp_Mh[0,i],qp_Mh[1,i]); Q10i = Q10(qp_Mh[0,i],qp_Mh[1,i])
            Q01i = Q01(qp_Mh[0,i],qp_Mh[1,i]); Q11i = Q11(qp_Mh[0,i],qp_Mh[1,i])
        
            for j in range(lphi_HDIV):
                phii_HDIV = phi_HDIV[j](qp_Mh[0,i],qp_Mh[1,i])
                phiix_HDIV[:,j] = 1/detQi*(Q00i*phii_HDIV[0] + Q01i*phii_HDIV[1])*DIRECTION_DOF[:,j]
                phiiy_HDIV[:,j] = 1/detQi*(Q10i*phii_HDIV[0] + Q11i*phii_HDIV[1])*DIRECTION_DOF[:,j]

            ellmatsM_HDIV_LUMPED = ellmatsM_HDIV_LUMPED + we_Mh[i]*(assem_ellmats(phiix_HDIV,phiix_HDIV) + assem_ellmats(phiiy_HDIV,phiiy_HDIV))*npy.abs(detQi)[:,None]
             
        ellmatsM_HDIV_LUMPED[abs(ellmatsM_HDIV_LUMPED)<1e-14] = 0
        MAT['Mh'] = sparse(im_HDIV,jm_HDIV,ellmatsM_HDIV_LUMPED,INFO['sizeM'],INFO['sizeM'])
        MAT['Mh'].eliminate_zeros()
    return MAT




def h1(MESH,BASIS,LISTS,Dict):
    space = Dict.get('space')
    time_elapsed = time.time()
    MAT = {};
    MAT_TRIG = {}; MAT_QUAD = {}
    
    t = MESH.t; q = MESH.q
    
    INFO = get_info(MESH,space)

    if t.shape[0] != 0: MAT_TRIG = h1_trig(MESH,BASIS,LISTS,INFO,Dict)
    if q.shape[0] != 0: MAT_QUAD = h1_quad(MESH,BASIS,LISTS,INFO,Dict)

    if t.shape[0] == 0: MAT = MAT_QUAD
    if q.shape[0] == 0: MAT = MAT_TRIG
    
    # MAT_BOUNDARY = h1_boundary(MESH,BASIS,LISTS,INFO)

    if t.shape[0] != 0 and q.shape[0] != 0:
        MAT_TRIG_Set = set(MAT_TRIG)
        MAT_QUAD_Set = set(MAT_QUAD)
        
        for field in MAT_TRIG_Set.intersection(MAT_QUAD_Set):
            # MAT[space][field] = MAT_TRIG[field] + MAT_QUAD[field]
            MAT[field] = MAT_TRIG[field] + MAT_QUAD[field]
            
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
        ellmatsM_H1 = npy.zeros((nt,lphi_H1*lphi_H1))
        
        for i in range(len(we_M)):
            qpT_i_1 = A00*qp_M[0,i]+A01*qp_M[1,i]+p[t0,0]
            qpT_i_2 = A10*qp_M[0,i]+A11*qp_M[1,i]+p[t0,1]
            coeff_qpT_i = coeff(qpT_i_1,qpT_i_2)
            for j in range(lphi_H1):
                phii_H1[:,j] = phi_H1[j](qp_M[0,i],qp_M[1,i])
    
            ellmatsM_H1 = ellmatsM_H1 + 1/2*we_M[i]*(assem_ellmats(phii_H1,phii_H1))*npy.abs(detA)[:,None]*coeff_qpT_i[:,None]*coeff_const[indices,None]
    
        ellmatsM_H1[abs(ellmatsM_H1)<1e-14] = 0
        M = sparse(im_H1,jm_H1,ellmatsM_H1,sizeM,sizeM)
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
        


def h1_quad(MESH,BASIS,LISTS,INFO,Dict): # TODO : übertrage h1_trig functionality
    p = MESH.p; np = MESH.np
    q = MESH.q; nq = MESH.nq
    
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


# def l2(MESH,BASIS,LISTS,Dict):
#     space = Dict.get('space')
#     time_elapsed = time.time()
#     MAT = {};
#     MAT_TRIG = {}; MAT_QUAD = {}
    
#     t = MESH.t; q = MESH.q
    
#     INFO = get_info(MESH,space)

#     if t.shape[0] != 0: MAT_TRIG = l2_trig(MESH,BASIS,LISTS,INFO,Dict)
#     # if q.shape[0] != 0: MAT_QUAD = l2_quad(MESH,BASIS,LISTS,INFO,Dict)

#     if t.shape[0] == 0: MAT = MAT_QUAD
#     # if q.shape[0] == 0: MAT = MAT_TRIG
    
#     # MAT_BOUNDARY = h1_boundary(MESH,BASIS,LISTS,INFO)

#     if t.shape[0] != 0 and q.shape[0] != 0:
#         MAT_TRIG_Set = set(MAT_TRIG)
#         MAT_QUAD_Set = set(MAT_QUAD)
        
#         for field in MAT_TRIG_Set.intersection(MAT_QUAD_Set):
#             # MAT[space][field] = MAT_TRIG[field] + MAT_QUAD[field]
#             MAT[field] = MAT_TRIG[field] + MAT_QUAD[field]
            
#     elapsed = time.time()-time_elapsed
#     print('Assembling ' + space + ' took ' + str(elapsed)[0:5] + ' seconds.')
#     return MAT


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