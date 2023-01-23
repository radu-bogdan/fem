import numpy as np
import pde
# from pde import *
import scipy.sparse as sps
import scipy.sparse.linalg
from sksparse.cholmod import cholesky
import time
from scikits import umfpack

import plotly.io as pio
pio.renderers.default = 'browser'
np.set_printoptions(precision = 8)
# np.set_printoptions(linewidth = 150)

# p,e,t,q = pde.petq_from_gmsh(filename = 'unit_square.geo',hmax = 1*1/np.sqrt(2)**5)
p,e,t,q = pde.petq_from_gmsh(filename = 'mesh_new.geo', hmax = 1/np.sqrt(2)**2)
MESH = pde.initmesh(p,e,t,q)
BASIS = pde.basis()
LISTS = pde.lists(MESH)

pex = lambda x,y : (1-x)**4+(1-y)**3*(1-x) + np.sin(1-y)*np.cos(1-x)
u1ex = lambda x,y : -np.sin(1-x)*np.sin(1-y)-4*(x-1)**3-(y-1)**3
u2ex = lambda x,y : -3*(x-1)*(y-1)**2 + np.cos(1-x)*np.cos(1-y)
divuex = lambda x,y : -6*(x-1)*(2*x+y-3)+2*np.cos(1-x)*np.sin(1-y)

# c1 = 0;
# c2 = 1;

# pex = lambda x,y : c1*x + c2*y
# u1ex = lambda x,y : -c1 + 0*x
# u2ex = lambda x,y : -c2 + 0*x
# divuex = lambda x,y : 0 + 0*x

# fig = MESH.pdemesh(info=0,border=0.6)
# fig.show()

# some projections

pex_l = pde.projections.evaluate(MESH, dict(trig = 'P0',quad = 'Q0'), pex)

divuex_P1d_Q0 = pde.projections.evaluate(MESH, dict(trig = 'P1d', quad = 'Q0'), divuex)
divuex_P1d_Q1d = pde.projections.evaluate(MESH, dict(trig = 'P1d', quad = 'Q1d'), divuex)
divuex_P0_Q0 = pde.projections.evaluate(MESH, dict(trig = 'P0', quad = 'Q0'), divuex)

M_divuex_P0_Q0_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig = 'P0', quad = 'Q0'), divuex)
M_divuex_P1d_Q0_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig = 'P1d', quad = 'Q0'), divuex)
M_divuex_P1d_Q1d_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig = 'P1d', quad = 'Q1d'), divuex)


M_u1ex_P1d_Q1d_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig = 'P1d', quad = 'Q1d'), u1ex)
M_u2ex_P1d_Q1d_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig = 'P1d', quad = 'Q1d'), u2ex)

uex_BDM1 = pde.projections.interp_HDIV(MESH, BASIS, 'BDM1', lambda x,y : np.c_[u1ex(x,y),u2ex(x,y)])

p_n_BDM1  = pde.projections.assem_HDIV_b(MESH, BASIS, dict(space = 'BDM1', edges = MESH.Boundary.TrigEdges, order = 2), pex)
p_n_BDM1 += pde.projections.assem_HDIV_b(MESH, BASIS, dict(space = 'BDM1', edges = MESH.Boundary.QuadEdges, order = 2), pex)

p_n_BDM1_inex  = pde.projections.assem_HDIV_b(MESH, BASIS, dict(space = 'BDM1', edges = MESH.Boundary.TrigEdges, order = 2), pex)
p_n_BDM1_inex += pde.projections.assem_HDIV_b(MESH, BASIS, dict(space = 'BDM1', edges = MESH.Boundary.QuadEdges, order = 0), pex)

MAT = {}
MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = 'RT0-RT0')
MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = 'BDM1-BDM1')
MAT = MAT | pde.assemble.h1(MESH,BASIS,LISTS,space = 'P1d-Q1d')


M = MAT['BDM1-BDM1']['M']
K = MAT['BDM1-BDM1']['K']
C = MAT['BDM1-BDM1']['C']
D = MAT['BDM1-BDM1']['D']
Mh = MAT['BDM1-BDM1']['Mh']

Mx_P1d_Q1d = MAT['BDM1-BDM1']['Mx_P1d_Q1d']
My_P1d_Q1d = MAT['BDM1-BDM1']['My_P1d_Q1d']
# M_ot = MAT['BDM1-BDM1']['Mh_only_trig']
# M_ot = MAT['BDM1-BDM1']['Mh']
M_P1d_Q1d = MAT['P1d-Q1d']['M']


A1 = sps.vstack((sps.hstack((Mh,-C.T)),
                 sps.hstack(( C, 0*D))))

A2 = sps.vstack((sps.hstack((M,-C.T)),
                 sps.hstack((C, 0*D))))

b1 = np.r_[-p_n_BDM1_inex,
            M_divuex_P0_Q0_new]

b2 = np.r_[-p_n_BDM1,
            M_divuex_P0_Q0_new]


cholMh = cholesky(Mh)

Afun = lambda x : (C@(cholMh.solve_A(C.T@x)))
b = M_divuex_P0_Q0_new - C@(cholMh.solve_A(-p_n_BDM1_inex))
ph3 = pde.pcg(Afun,b,maxit=1e10,tol=1e-15)
uh3 = cholMh.solve_A(C.T@ph3-p_n_BDM1_inex)


# t = time.time()
# u_inter = sps.linalg.spsolve(C@C.T)


t = time.time()
res1 = sps.linalg.spsolve(A1,b1)
res2 = sps.linalg.spsolve(A2,b2)
print("Took",str(time.time()-t)[:5], "seconds.")

uh1 = res1[:2*MESH.NoEdges]
uh2 = res2[:2*MESH.NoEdges]

ph1 = res1[2*MESH.NoEdges:]
ph2 = res2[2*MESH.NoEdges:]


# xy = sps.linalg.spsolve(M,M[:,0])

print(np.linalg.norm(ph1-ph3,np.inf) + np.linalg.norm(uh1-uh3,np.inf))


exit()


uh1r = uh1.reshape((uh1.size//2,2))
uh1r = np.mean(uh1r,1)
uh1r = np.tile(uh1r, (2,1)).flatten('F')
uh1_old = uh1
uh1 = uh1r

# A3 = sps.vstack((sps.hstack((M,-C.T)),
#                  sps.hstack((C, 0*D))))

# cholMh = cholesky(Mh)


# uh1 = 0*uh1
# uh1[159*2] = 1


# b_new = np.r_[Mh*uh1,
#                C*uh1]

# res3 = sps.linalg.spsolve(A3,b_new)

for edge in MESH.Lists.InterfaceTriangleQuad:
    # print(elem)
    # print()
    trig = np.argwhere(MESH.TriangleToEdges == edge)[0][0]
    edges_trig = MESH.TriangleToEdges[trig,:]
    index = np.argwhere(edges_trig == edge)[0][0]
    list_trig = LISTS['BDM1']['TRIG']['LIST_DOF'][trig,:]
    
    print(trig)
    print(edges_trig)
    print(index)
    print(list_trig)
    
    if index == 0:
        edge_to_mod_1 = list_trig[2]; edge_from_mod_1 = list_trig[3]
        edge_to_mod_2 = list_trig[5]; edge_from_mod_2 = list_trig[4]
        
    if index == 1:
        edge_to_mod_1 = list_trig[1]; edge_from_mod_1 = list_trig[0]
        edge_to_mod_2 = list_trig[4]; edge_from_mod_2 = list_trig[5]
        
    if index == 2:
        edge_to_mod_1 = list_trig[0]; edge_from_mod_1 = list_trig[1]
        edge_to_mod_2 = list_trig[3]; edge_from_mod_2 = list_trig[2]
    
    
    print(str(edge_to_mod_1)+ " und " + str(edge_to_mod_2))
    print('\n')
    
    # uh1[int(edge_to_mod_1)] = uh1[int(edge_from_mod_1)]
    # uh1[int(edge_to_mod_2)] = uh1[int(edge_from_mod_2)]
    
    
    # print(trig)
    
    

###############################################################################


for i in range(len(MESH.Lists.InterfaceTriangleQuad)):
    edge = MESH.Lists.InterfaceTriangleQuad[i]
    print(edge)
    t_patch = np.argwhere(MESH.TriangleToEdges == edge)[0][0]
    q_patch = np.argwhere(MESH.QuadToEdges == edge)[0][0]
    
    ie = np.r_[2*edge, 2*edge+1]
    ie1 = np.r_[2*edge] 
    
    M_patch = M[ie,:]; M_patch = M_patch[:,ie].toarray()
    C_patch = C[:,ie]; C_patch = C_patch[np.r_[t_patch],:].toarray()
    # C_patch = C[:,ie]; C_patch = C_patch[np.r_[t_patch,MESH.nt + q_patch],:].toarray()
    Z_patch = sps.csr_matrix((1, 1), dtype=np.int64).toarray()
    
    it = MESH.TriangleToEdges[t_patch,:]; iq = MESH.QuadToEdges[q_patch,:]
    ito = np.setdiff1d(it,edge); iqo = np.setdiff1d(iq,edge)
    ii = np.union1d(ito,iqo); iii = np.r_[2*ii, 2*ii+1]
    
    M_rhs = M[ie,:]; M_rhs = M_rhs[:,iii].toarray()
    # C_rhs = C[:,edge]; C_rhs = C_rhs[np.r_[t_patch,MESH.nt + q_patch],:].toarray()
    C_rhs = C[np.r_[t_patch],:]; C_rhs = C_rhs[:,iii].toarray()
    
    print(C_patch@uh1[ie])
    
    A_loc = np.r_[np.c_[M_patch,-C_patch.T],
                  np.c_[C_patch, Z_patch]]
    
    r_loc = np.r_[-M_rhs@uh1[iii],
                   C_patch@uh1[ie]]
    
    # r_loc = np.r_[M_rhs@uh1[iii],
    #               -C_rhs@uh1[iii] + C_rhs@uh2[iii]+C_patch@uh2[ie]]
    
    res_loc = sps.linalg.spsolve(A_loc,r_loc)
    
    # uh1[ie] = res_loc[0:2]
    
    
    print(C_patch@uh1[ie])
    print("\n")

###############################################################################



uh1_x_l = sps.linalg.spsolve(M_P1d_Q1d,Mx_P1d_Q1d*uh1)
uh1_y_l = sps.linalg.spsolve(M_P1d_Q1d,My_P1d_Q1d*uh1)

uh2_x_l = sps.linalg.spsolve(M_P1d_Q1d,Mx_P1d_Q1d*uh2)
uh2_y_l = sps.linalg.spsolve(M_P1d_Q1d,My_P1d_Q1d*uh2)

# uh3_x_l = sps.linalg.spsolve(M_P1d_Q1d,Mx_P1d_Q1d*uh3)
# uh3_y_l = sps.linalg.spsolve(M_P1d_Q1d,My_P1d_Q1d*uh3)

uex_x_l = sps.linalg.spsolve(M_P1d_Q1d,Mx_P1d_Q1d*uex_BDM1)
uex_y_l = sps.linalg.spsolve(M_P1d_Q1d,My_P1d_Q1d*uex_BDM1)

# uex_x_l = sps.linalg.spsolve(M_P1d_Q1d,M_u1ex_P1d_Q1d_new)
# uex_y_l = sps.linalg.spsolve(M_P1d_Q1d,M_u2ex_P1d_Q1d_new)


t = time.time()
fig = MESH.pdesurf_hybrid(dict(trig = 'P1d',quad = 'Q1d', controls = 1),uh1_x_l)
# fig = MESH.pdesurf_hybrid(dict(trig = 'P1d',quad = 'Q1d', controls = 1),np.abs(uh1_x_l-uh2_x_l) + np.abs(uh1_y_l-uh2_y_l))
# fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1),np.abs(ph1-ph2))
# fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1), ph1-ph2)
print("Took",str(time.time()-t)[:5], "seconds.")

t = time.time()
fig.show()
print("Plotting took",str(time.time()-t)[:5], "seconds.")


from matplotlib.pyplot import plot
# plot(np.abs(uh1-uh2))