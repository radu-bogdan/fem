
import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
from sksparse.cholmod import cholesky


import plotly.io as pio
pio.renderers.default = 'browser'

# p,e,t,q = pde.petq_from_gmsh(filename = 'mesh_new.geo',hmax = 0.180)
p,e,t,q = pde.petq_from_gmsh(filename = 'mesh_new.geo',hmax = 0.750)
print(q.dtype)
print(t.dtype)

MESH = pde.initmesh(p,e,t,q)

print(MESH.q.dtype)
print(MESH.t.dtype)

BASIS = pde.basis()
LISTS = pde.lists(MESH)

pex = lambda x,y : (1-x)**4+(1-y)**3*(1-x) + np.sin(1-y)*np.cos(1-x)
u1ex = lambda x,y : -np.sin(1-x)*np.sin(1-y)-4*(x-1)**3-(y-1)**3
u2ex = lambda x,y : -3*(x-1)*(y-1)**2 + np.cos(1-x)*np.cos(1-y)
divuex = lambda x,y : -6*(x-1)*(2*x+y-3)+2*np.cos(1-x)*np.sin(1-y)


# pex = lambda x,y : y
# u1ex = lambda x,y :  0+0*x
# u2ex = lambda x,y : -1+0*x
# divuex = lambda x,y : 0+0*x


# fig = MESH.pdemesh(info=0,border=0.6)
# fig.show()

# some projections

pex_l = pde.projections.evaluate(MESH, dict(trig = 'P0',quad = 'Q0'), pex)

divuex_P1d_Q0 = pde.projections.evaluate(MESH, dict(trig = 'P1d',quad = 'Q0'), divuex)
divuex_P1d_Q1d = pde.projections.evaluate(MESH, dict(trig = 'P1d',quad = 'Q1d'), divuex)
divuex_P0_Q0 = pde.projections.evaluate(MESH, dict(trig = 'P0',quad = 'Q0'), divuex)

M_divuex_P0_Q0_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig = 'P0',quad = 'Q0'), divuex)
M_divuex_P1d_Q0_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig = 'P1d',quad = 'Q0'), divuex)
M_divuex_P1d_Q1d_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig = 'P1d',quad = 'Q1d'), divuex)


M_u1ex_P1d_Q1d_new = pde.projections.assem(MESH, BASIS, dict(trig = 'P1d',quad = 'Q1d'), u1ex)
M_u2ex_P1d_Q1d_new = pde.projections.assem(MESH, BASIS, dict(trig = 'P1d',quad = 'Q1d'), u2ex)

# uex is noch falsch ...
uex_BDM1 = pde.projections.interp_HDIV(MESH, BASIS, 'BDM1', lambda x,y : np.c_[u1ex(x,y),u2ex(x,y)])

p_n_BDM1  = pde.projections.assem_HDIV_b(MESH, BASIS, dict(space = 'BDM1', edges = MESH.Boundary.TrigEdges, order = 2), pex)
p_n_BDM1 += pde.projections.assem_HDIV_b(MESH, BASIS, dict(space = 'BDM1', edges = MESH.Boundary.QuadEdges, order = 0), pex)


MAT = {}
MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = 'RT0-RT0')
MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = 'BDM1-BDM1')
MAT = MAT | pde.assemble.h1(MESH,BASIS,LISTS,space = 'P1d-Q1d')



M = MAT['BDM1-BDM1']['M']
Mx_P1d_Q1d = MAT['BDM1-BDM1']['Mx_P1d_Q1d']
My_P1d_Q1d = MAT['BDM1-BDM1']['My_P1d_Q1d']
K = MAT['BDM1-BDM1']['K']
C = MAT['BDM1-BDM1']['C']
Mh = MAT['BDM1-BDM1']['Mh']
M_P1d_Q1d = MAT['P1d-Q1d']['M']

Z = sps.coo_matrix((C.shape[0],C.shape[0]), dtype = np.float64)

A1 = sps.vstack((sps.hstack((Mh,-C.T)),
                 sps.hstack(( C,   Z))))
A2 = sps.vstack((sps.hstack((M,-C.T)),
                 sps.hstack(( C,   Z))))


b = np.r_[-p_n_BDM1,
           M_divuex_P0_Q0_new]









from scikits import umfpack
from scipy.sparse.linalg import splu
cholMh = cholesky(Mh)

# spluA = splu(Mh)
# print(spluA.perm_c.shape)
# print(scipy.sparse.linalg.norm(spluA.L@spluA.U-Mh))
print(scipy.sparse.linalg.norm(cholMh.L()@cholMh.L().T-Mh[cholMh.P()[:, np.newaxis], cholMh.P()[np.newaxis, :]]))
# print(cholMh.P().shape)

from scipy.sparse.linalg import (inv, spsolve)

t = time.time()
spluL = umfpack.splu(cholMh.L().T)
R = spluL.solve(C.T)
CT = C.T



# iL = cholMh(C.T)


# res = 0*CT

# for i in range(Mh.shape[0]):
#     res[i,:] = CT[i,:]
#     for j in range(i):
#         res[i,:] -= Mh[i,j]*res[j,:]
#     res[i,:] /= Mh[i,i]


# xx = scipy.sparse.linalg.spsolve_triangular(cholMh.L(),CT.todense())
# iL = inv(cholMh.L())

# print(C.T.shape)

# LL = cholMh.L().tocsc()
print(time.time()-t)

t = time.time()
# ans = spluL.solve_sparse(C.T)
# ans = inv(cholMh.L())
# print(cholMh.L().shape)
# lu = umfpack.splu(cholMh.L())
# x = umfpack.spsolve(LL, C.T)
# xx = scipy.sparse.linalg.spsolve(LL, C.T, permc_spec='COLAMD', use_umfpack=True)

# lu.solve_sparse(C.T)

print(time.time()-t)
                        
from matplotlib.pyplot import spy
spy(cholMh.L()@cholMh.L().T,markersize = 0.5)
# spy(cholMh.L(),markersize = 0.01)
print(C.T.nnz)
# print(iL)


