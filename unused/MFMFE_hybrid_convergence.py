iter = 9

import numpy as np; # np.set_printoptions(precision = 8)
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import sys
print(sys.path)

#import sys
#!{sys.executable} -m pip install scikit-sparse

from sksparse.cholmod import cholesky
# from scipy.sparse.linalg import inv as sinv
# from scikits.umfpack import spsolve, splu

pex = lambda x,y : (1-x)**4+(1-y)**3*(1-x) + np.sin(1-y)*np.cos(1-x)
u1ex = lambda x,y : -np.sin(1-x)*np.sin(1-y)-4*(x-1)**3-(y-1)**3
u2ex = lambda x,y : -3*(x-1)*(y-1)**2 + np.cos(1-x)*np.cos(1-y)
divuex = lambda x,y : -6*(x-1)*(2*x+y-3)+2*np.cos(1-x)*np.sin(1-y)

# pex = lambda x,y : x+y
# u1ex = lambda x,y : -1+0*x
# u2ex = lambda x,y : -1+0*x
# divuex = lambda x,y : 0+0*x


error_uh = np.empty(iter)
error_ph = np.empty(iter)

geo = 'mesh_new.geo'
# geo = 'unit_square.geo'

for i in range(iter):
    
    print("Iteration",str(i),"/",str(iter-1))
    p,e,t,q = pde.petq_from_gmsh(filename = geo,hmax = 1*1/np.sqrt(2)**i)
    
    MESH = pde.initmesh(p,e,t,q); BASIS = pde.basis(); LISTS = pde.lists(MESH)
    
    MAT = {}
    MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = 'BDM1-BDM1')
    MAT = MAT | pde.assemble.h1(MESH,BASIS,LISTS,space = 'P1d-Q1d')
    
    p_n_BDM1  = pde.projections.assem_HDIV_b(MESH, BASIS, dict(space = 'BDM1', edges = MESH.Boundary.TrigEdges, order = 2), pex)
    p_n_BDM1 += pde.projections.assem_HDIV_b(MESH, BASIS, dict(space = 'BDM1', edges = MESH.Boundary.QuadEdges, order = 0), pex)

    M_divuex_P0_Q0_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig = 'P0',quad = 'Q0'), divuex)
    
    M = MAT['BDM1-BDM1']['M']
    C = MAT['BDM1-BDM1']['C']
    D = MAT['BDM1-BDM1']['D']
    Mh = MAT['BDM1-BDM1']['Mh']
    Mx_P1d_Q1d = MAT['BDM1-BDM1']['Mx_P1d_Q1d']
    My_P1d_Q1d = MAT['BDM1-BDM1']['My_P1d_Q1d']
    M_P1d_Q1d = MAT['P1d-Q1d']['M']
    
    Z = sps.coo_matrix((C.shape[0],C.shape[0]), dtype = np.float64)
    
    # A = sps.vstack((sps.hstack((Mh,-C.T)),
    #                 sps.hstack((C,   Z))))
    
#    b = np.r_[-p_n_BDM1,
#               M_divuex_P0_Q0_new]
    
    cholMh = cholesky(Mh)
    
    Afun = lambda x : (C@(cholMh.solve_A(C.T@x)))
    b = M_divuex_P0_Q0_new - C@(cholMh.solve_A(-p_n_BDM1))
    ph = pde.pcg(Afun,b,maxit=1e10,tol=1e-15)
    uh = cholMh.solve_A(C.T@ph-p_n_BDM1)
    
    
    # res = sps.linalg.spsolve(A,b)
    
    # uh = res[:2*MESH.NoEdges]; ph = res[2*MESH.NoEdges:]
    
    uex_BDM1 = pde.projections.interp_HDIV(MESH, BASIS, 'BDM1', lambda x,y : np.c_[u1ex(x,y),u2ex(x,y)])
    pex_P0 = pde.projections.evaluate(MESH, dict(trig = 'P0',quad = 'Q0'), pex)
    
#     uhr = uh.reshape((uh.size//2,2))
#     uhr = np.mean(uhr,1)
#     uhr = np.tile(uhr, (2,1)).flatten('F')
#     uh_old = uh
#     uh = uhr
    
    
    error_uh[i] = np.sqrt((uh-uex_BDM1)@M@(uh-uex_BDM1))
    error_ph[i] = np.sqrt((ph-pex_P0)@D@(ph-pex_P0))
    
rate = np.log2(error_uh[0:-1]/error_uh[1:])
print(rate)

rate = np.log2(error_ph[0:-1]/error_ph[1:])
print(rate)    