import numpy as npy
from . import quadrature
from .assemble import assem_ellmats
from .basis import basis


# ###############################################################################

# def evaluateP1_trig(MESH,Dict,U):
#     if 'regions' in Dict.keys():
#         regions = Dict.get('regions')
#     else:
#         regions = MESH.RegionsT
    
#     indices = npy.unique(MESH.RegionsToPoints[npy.in1d(MESH.RegionsToPoints[:,0],regions),1])
#     real_indices = npy.in1d(npy.r_[0:MESH.np],indices)
    
#     return U(MESH.p[:,0],MESH.p[:,1])*real_indices

# def evaluateP0_trig(MESH,Dict,U):
#     if 'regions' in Dict.keys():
#         regions = Dict.get('regions')
#     else:
#         regions = MESH.RegionsT
    
#     # indices = npy.argwhere(npy.in1d(MESH.RegionsT,regions))[:,0]
    
#     indices = npy.argwhere(npy.in1d(MESH.t[:,3],regions))[:,0]
#     real_indices = npy.in1d(npy.r_[0:MESH.nt],indices)
    
#     xx = 1/3*(MESH.p[MESH.t[:,0],0] + MESH.p[MESH.t[:,1],0] + MESH.p[MESH.t[:,2],0])
#     yy = 1/3*(MESH.p[MESH.t[:,0],1] + MESH.p[MESH.t[:,1],1] + MESH.p[MESH.t[:,2],1])
    
#     return U(xx,yy)*real_indices

# def evaluate(MESH,Dict,U):
    
#     trig = Dict.get('trig')
#     quad = Dict.get('quad')
#     if 'regions' in Dict.keys():
#         regions = Dict.get('regions')
#     else:
#         regions = MESH.RegionsT
    
#     indices = npy.unique(MESH.RegionsToPoints[npy.in1d(MESH.RegionsToPoints[:,0],regions),1])
#     real_indices = npy.in1d(npy.r_[0:MESH.np],indices)
    
#     if trig == 'P1' or quad == 'Q1':
        
#         return U(MESH.p[:,0],MESH.p[:,1])*real_indices
#     else:
#         if MESH.t.shape[0]!=0:
#             U_l_1_trig = __evaluate_trig(MESH,trig,U,real_indices)
#         else:
#             U_l_1_trig = npy.array([],dtype=npy.uint64)
        
#         if MESH.q.shape[0]!=0:
#             U_l_1_quad = __evaluate_quad(MESH,quad,U,real_indices)
#         else:
#             U_l_1_quad = npy.array([],dtype=npy.uint64)
    
#         return npy.r_[U_l_1_trig.flatten(),U_l_1_quad.flatten()]


# def __evaluate_trig(MESH,trig,U,real_indices):
#     U_l_1 = U(MESH.p[:,0],MESH.p[:,1])*real_indices
    
#     U_l_1_P1d = npy.c_[U_l_1[MESH.t[:,0]],
#                        U_l_1[MESH.t[:,1]],
#                        U_l_1[MESH.t[:,2]]]
    
#     if trig == 'P1d':
#         return U_l_1_P1d
    
#     if trig == 'P0':
#         return U_l_1_P1d.mean(axis = 1)
    

# def __evaluate_quad(MESH,quad,U,real_indices):
#     U_l_1 = U(MESH.p[:,0],MESH.p[:,1])*real_indices
    
#     U_l_1_Q1d = npy.c_[U_l_1[MESH.q[:,0]],
#                        U_l_1[MESH.q[:,1]],
#                        U_l_1[MESH.q[:,2]],
#                        U_l_1[MESH.q[:,3]]]
    
#     if quad == 'Q1d':
#         return U_l_1_Q1d
    
#     if quad == 'Q0':
#         return U_l_1_Q1d.mean(axis = 1)
# ###############################################################################






# ###############################################################################
# def assemH1(MESH, BASIS, LISTS, Dict, U):
    
#     if MESH.q.shape[0]!=0:
#         uh_quad = __assem_quadH1(MESH, BASIS, LISTS, Dict, U)
#     else:
#         uh_quad = npy.array([],dtype=npy.uint64)
    
#     if MESH.t.shape[0]!=0:
#         uh_trig = __assem_trigH1(MESH, BASIS, LISTS, Dict, U)
#     else:
#         uh_trig = npy.array([],dtype=npy.uint64)
    
#     uh = npy.r_[uh_trig,uh_quad]
    
#     return uh
    
# def __assem_trigH1(MESH, BASIS, LISTS, Dict, U):
    
#     trig = Dict.get('trig')
    
#     if 'regions' in Dict.keys():
#         regions = Dict.get('regions')
#     else:
#         regions = MESH.RegionsT
    
#     indices = npy.argwhere(npy.in1d(MESH.RegionsT,regions))[:,0]
    
#     phi = BASIS[trig]['TRIG']['phi']
#     liste = LISTS[trig]['TRIG']['LIST_DOF']
#     sizeM = int(npy.max(liste))+1 # TODO : fix this
    
#     liste_cut = liste[indices,:]
    
#     p = MESH.p; t = MESH.t[indices,:]; nt = t.shape[0];
#     qp,we = quadrature.dunavant(order = 2)
    
#     lphi = len(phi)
#     phii = npy.zeros((nt,lphi))
    
#     t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]
#     A00 = p[t1,0]-p[t0,0]; A01 = p[t2,0]-p[t0,0]
#     A10 = p[t1,1]-p[t0,1]; A11 = p[t2,1]-p[t0,1]
#     detA = A00*A11-A01*A10
    
#     ellmatsT = npy.zeros((nt,lphi))
    
#     for i in range(len(we)):
#         qpT_i_1 = A00*qp[0,i]+A01*qp[1,i]+p[t0,0]
#         qpT_i_2 = A10*qp[0,i]+A11*qp[1,i]+p[t0,1]
        
#         u_qpT_i = U(qpT_i_1,qpT_i_2)
        
#         for j in range(lphi):
#             phii[:,j] = phi[j](qp[0,i],qp[1,i])
            
#         ellmatsT = ellmatsT + 1/2*we[i]*assem_ellmats(u_qpT_i, phii)*npy.abs(detA)[:,None]
        
#     # conversion from uint to int needed by bincount... apparently
#     uh = npy.bincount(liste_cut.flatten().astype(int), weights = ellmatsT.flatten(), minlength = sizeM)
#     # uh = npy.bincount(liste.flatten().astype(int), weights = ellmatsT.flatten(),minlength = ANGEBEN!)
#     return uh



# def __assem_quadH1(MESH, BASIS, LISTS, Dict, U):
    
#     quad = Dict.get('quad')
    
#     if quad == 'Q0':
#         phi = BASIS['Q0']['QUAD']['phi']
#         liste = LISTS['Q0']['QUAD']['LIST_DOF']
#     if quad == 'Q1d':
#         phi = BASIS['Q1']['QUAD']['phi']
#         liste = LISTS['Q1d']['QUAD']['LIST_DOF']
#     if quad == 'P1d':
#         phi = BASIS['P1']['QUAD']['phi']
#         liste = LISTS['P1d']['QUAD']['LIST_DOF']
    
#     p = MESH.p; q = MESH.q; nq = MESH.nq;
#     qp,we = quadrature.quadrule(order = 5)
    
#     lphi = len(phi)
#     phii = npy.zeros((nq,lphi))
    
#     q0 = q[:,0]; q1 = q[:,1]; q2 = q[:,2]; q3 = q[:,3]
    
#     B00 = p[q1,0]-p[q0,0]; B01 = p[q3,0]-p[q0,0]
#     B10 = p[q1,1]-p[q0,1]; B11 = p[q3,1]-p[q0,1]
#     detB = abs(B00*B11-B01*B10)
    
#     C00 = p[q2,0]-p[q3,0]; C01 = p[q3,0]-p[q0,0]
#     C10 = p[q2,1]-p[q3,1]; C11 = p[q3,1]-p[q0,1]
#     detC = abs(C00*C11-C01*C10)
    
#     D00 = p[q1,0]-p[q0,0]; D01 = p[q2,0]-p[q1,0]
#     D10 = p[q1,1]-p[q0,1]; D11 = p[q2,1]-p[q1,1]
#     detD = abs(D00*D11-D01*D10)
    
#     detQ = lambda x,y: detB + (detD-detB)*x + (detC-detB)*y
    
#     R0 = p[q2,0]-p[q3,0]-p[q1,0]+p[q0,0]
#     R1 = p[q2,1]-p[q3,1]-p[q1,1]+p[q0,1]
    
#     Q00 = lambda x,y : B00+R0*y; Q01 = lambda x,y : B01+R1*x
#     Q10 = lambda x,y : B10+R0*y; Q11 = lambda x,y : B11+R1*x # not 100% sure on this one ...    
    
#     ellmatsT = npy.zeros((nq,lphi))
    
#     for i in range(len(we)):
#         detQi = detQ(qp[0,i],qp[1,i])
#         qpT_i_1 = Q00(qp[0,i],qp[1,i])*qp[0,i] + Q01(qp[0,i],qp[1,i])*qp[1,i] + p[q0,0]
#         qpT_i_2 = Q10(qp[0,i],qp[1,i])*qp[0,i] + Q11(qp[0,i],qp[1,i])*qp[1,i] + p[q0,1]
        
#         u_qpT_i = U(qpT_i_1,qpT_i_2)
        
#         for j in range(lphi):
#             phii[:,j] = phi[j](qp[0,i],qp[1,i])
        
#         ellmatsT = ellmatsT + we[i]*assem_ellmats(u_qpT_i[:,None], phii)*npy.abs(detQi)[:,None]
        
#     uh = npy.bincount(liste.flatten().astype(int), weights = ellmatsT.flatten())
#     return uh
# ###############################################################################






# ###############################################################################
# def assem_HDIV_b(MESH, BASIS, Dict, U):
#     space = Dict.get('space')
#     edges = Dict.get('edges')
#     order = Dict.get('order')
    
#     if 'edges' in Dict.keys():
#         edges = Dict.get('edges')
#     else:
#         edges = MESH.Boundary_Edges
    
#     indices = npy.nonzero(npy.in1d(MESH.Boundary_Edges,edges))[0]
    
#     p = MESH.p; e = MESH.e;
    
#     qp,we = quadrature.one_d(order)
    
#     e0 = e[:,0]; e1 = e[:,1]
#     A0 =  p[e1,0]-p[e0,0]; A1 = p[e1,1]-p[e0,1]
    
#     phi = {}
#     if space == 'RT0':
#         phi[0] = lambda x : 1 + 0*x
#         dofs = MESH.Boundary_Edges
    
#     if space == 'BDM1':
#         phi[0] = lambda x : 1-x
#         phi[1] = lambda x : x
#         dofs = npy.c_[2*MESH.Boundary_Edges,
#                       2*MESH.Boundary_Edges + 1]
        
#     lphi = len(phi)
    
#     phii = npy.zeros((MESH.Boundary_NoEdges,lphi))
#     ellmatsT = npy.zeros((MESH.Boundary_NoEdges,lphi))
    
#     uh = npy.zeros(lphi*MESH.NoEdges)
    
#     for i in range(len(we)):
#         qpT_i_1 = A0*qp[i] + p[e0,0]
#         qpT_i_2 = A1*qp[i] + p[e0,1]
        
#         u_qpT_i = U(qpT_i_1,qpT_i_2)
        
#         for j in range(lphi):
#             phii[:,j] = phi[j](qp[i])
        
#         ellmatsT = ellmatsT + we[i]*assem_ellmats(u_qpT_i[:,None], phii*MESH.Boundary_EdgeOrientation[:,None])
    
    
#     ellmatsT = ellmatsT[indices,:]
#     dofs = dofs[indices,:]
    
#     uh[dofs.flatten()] = ellmatsT.flatten()
#     # uh = npy.bincount(dofs.flatten().astype(int), weights = ellmatsT.flatten(),minlength = size)
#     return uh
# ###############################################################################




# ###############################################################################
# def assem_H1_b(MESH, BASIS, LISTS, Dict, U):
#     space = Dict.get('space')
#     edges = Dict.get('edges')
#     order = Dict.get('order')
#     size = Dict.get('size')
    
#     if 'edges' in Dict.keys():
#         edges = Dict.get('edges')
#     else:
#         edges = MESH.Boundary.Region
    
#     indices = npy.argwhere(npy.in1d(MESH.Boundary.Region,edges))[:,0]
    
#     p = MESH.p;    
#     e = MESH.e[indices,:]; ne = e.shape[0]
    
#     qp,we = quadrature.one_d(order)
    
#     e0 = e[:,0]; e1 = e[:,1]
#     A0 =  p[e1,0]-p[e0,0]; A1 =  p[e1,1]-p[e0,1]
#     detA = npy.sqrt(A0**2+A1**2)
        
#     phi = BASIS[space]['B']['phi']
#     dofs = LISTS[space]['B']['LIST_DOF'][indices,:]
        
#     lphi = len(phi)
    
#     phii = npy.zeros((ne,lphi))
#     ellmatsT = npy.zeros((ne,lphi))
    
#     uh = npy.zeros(size);
    
#     for i in range(len(we)):
#         qpT_i_1 = A0*qp[i] + p[e0,0]
#         qpT_i_2 = A1*qp[i] + p[e0,1]
        
#         u_qpT_i = U(qpT_i_1,qpT_i_2)
        
#         for j in range(lphi):
#             phii[:,j] = phi[j](qp[i])
        
#         ellmatsT = ellmatsT + we[i]*assem_ellmats(u_qpT_i[:,None], phii)*npy.abs(detA)[:,None]
    
#     # uh[dofs.flatten()] = ellmatsT.flatten()
#     uh = npy.bincount(dofs.flatten().astype(int), weights = ellmatsT.flatten(),minlength = size)
#     return uh
# ###############################################################################


from pde.hdiv.spaces import spaceInfo

###############################################################################
def interp_HDIV(MESH, space, order, f):
    
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH, space)
        
    p = MESH.p; # t = MESH.t; nt = MESH.nt;
    
    qp,we = quadrature.one_d(order)
    
    e0 = MESH.EdgesToVertices[:,1]; e1 = MESH.EdgesToVertices[:,0]
    A0 =  p[e1,0]-p[e0,0]; A1 =  p[e1,1]-p[e0,1]
    
    scaled_normal = npy.c_[-A1,A0] # scales so we don't need any determinants in the integral below, outstanding, kekw.
    
    phi =  MESH.FEMLISTS[space]['TRIG']['phidual']; lphi = len(phi)
    
    phii = npy.zeros((MESH.NoEdges,lphi))
    ellmatsT = npy.zeros((MESH.NoEdges,lphi))
    
    # uh = npy.zeros((MESH.NoEdges,1))
    
    for i in range(len(we)):
        qpT_i_1 = A0*qp[i] + p[e0,0]
        qpT_i_2 = A1*qp[i] + p[e0,1]
        
        u_qpT_i = f(qpT_i_1,qpT_i_2)
        
        for j in range(lphi):
            phii[:,j] = phi[j](qp[i])
        ellmatsT = ellmatsT + we[i]*assem_ellmats(npy.sum(u_qpT_i*scaled_normal,axis=1)[:,None], phii)
    
    return ellmatsT.flatten()
###############################################################################




###############################################################################
def interp_HDIV(MESH, space, order, f):
    
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH, space)
        
    p = MESH.p; # t = MESH.t; nt = MESH.nt;
    
    qp,we = quadrature.one_d(order)
    
    e0 = MESH.EdgesToVertices[:,1]; e1 = MESH.EdgesToVertices[:,0]
    A0 =  p[e1,0]-p[e0,0]; A1 =  p[e1,1]-p[e0,1]
    
    scaled_normal = npy.c_[-A1,A0] # scales so we don't need any determinants in the integral below, outstanding, kekw.
    
    phi =  MESH.FEMLISTS[space]['TRIG']['phidual']; lphi = len(phi)
    
    phii = npy.zeros((MESH.NoEdges,lphi))
    ellmatsT = npy.zeros((MESH.NoEdges,lphi))
    
    # uh = npy.zeros((MESH.NoEdges,1))
    
    for i in range(len(we)):
        qpT_i_1 = A0*qp[i] + p[e0,0]
        qpT_i_2 = A1*qp[i] + p[e0,1]
        
        u_qpT_i = f(qpT_i_1,qpT_i_2)
        
        for j in range(lphi):
            phii[:,j] = phi[j](qp[i])
        ellmatsT = ellmatsT + we[i]*assem_ellmats(npy.sum(u_qpT_i*scaled_normal,axis=1)[:,None], phii)
    
    return ellmatsT.flatten()
###############################################################################























