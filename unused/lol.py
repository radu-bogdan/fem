import pde
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from sksparse.cholmod import cholesky
import time

# from sys import exit
# from scipy import sparse, io, integrate, signal
# from scipy.sparse import linalg
# from scikits.umfpack import spsolve, splu, UmfpackLU

import plotly.io as pio
pio.renderers.default = 'browser'

np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = np.inf)
np.set_printoptions(precision = 3)

p,e,t,q = pde.petq_from_gmsh(filename = 'mesh_new.geo',hmax = 0.25)
# p,e,t,q = pde.petq_from_gmsh(filename = 'mesh_new.geo',hmax = 0.70711)



MESH = pde.mesh(p,e,t,q)
BASIS = pde.basis()
LISTS = pde.lists(MESH)

MAT = {}
# MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = 'RT0-RT0')
MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = 'BDM1-BDM1')
# MAT = MAT | pde.assemble.h1(MESH,BASIS,LISTS,space = 'P1-Q1')
MAT = MAT | pde.assemble.h1(MESH,BASIS,LISTS,dict(space = 'P1d-Q1d'))
# MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = 'RT1-BDFM1')
# MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = 'EJ1-RT0')

# TODO : dsa
# A = 'BDM1-RT0'
# A.split('-')[0]

# MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = dict(trig = 'BDM1',quad = 'BDM1'))

# space = 'RT1-BDFM1'
space = 'BDM1-BDM1'
# space = 'RT0-RT0'

M = MAT[space]['M']
Mx_P1d = MAT[space]['Mx_P1d_Q1d']
My_P1d = MAT[space]['My_P1d_Q1d']
K = MAT[space]['K']
C = MAT[space]['C']
Mh = MAT[space]['Mh']

M_P1d_Q1d = MAT['P1d-Q1d']['M']

###############################################################################
pex = lambda x,y : (1-x)**4+(1-y)**3*(1-x) + np.sin(1-y)*np.cos(1-x)
u1ex = lambda x,y : -np.sin(1-x)*np.sin(1-y)-4*(x-1)**3-(y-1)**3
u2ex = lambda x,y : -3*(x-1)*(y-1)**2 + np.cos(1-x)*np.cos(1-y)
divuex = lambda x,y : -6*(x-1)*(2*x+y-3)+2*np.cos(1-x)*np.sin(1-y)
###############################################################################



pex_l = pde.projections.evaluate(MESH, dict(trig='P1d',quad='Q1d'), pex)

divuex_P1d_Q0 = pde.projections.evaluate(MESH, dict(trig='P1d',quad='Q0'), divuex)
divuex_P1d_Q1d = pde.projections.evaluate(MESH, dict(trig='P1d',quad='Q1d'), divuex)
divuex_P0_Q0 = pde.projections.evaluate(MESH, dict(trig='P0',quad='Q0'), divuex)

M_divuex_P0_Q0_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig='P0',quad='Q0'), divuex)
M_divuex_P1d_Q0_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig='P1d',quad='Q0'), divuex)
M_divuex_P1d_P1d_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig='P1d',quad='P1d'), divuex)
M_divuex_P1d_Q1d_new = pde.projections.assem(MESH, BASIS, LISTS, dict(trig='P1d',quad='Q1d'), divuex)

# M_divuex_P0_Q0 = pde.projections.assem_P0_Q0(MESH, divuex)
# M_divuex_P1d_Q1d = pde.projections.assem_P1d_Q1d(MESH, divuex)

uex_BDM1 = pde.projections.interp_HDIV(MESH, BASIS, 'BDM1', lambda x,y : np.c_[u1ex(x,y),u2ex(x,y)])
p_n_BDM1 = pde.projections.assem_HDIV_b(MESH, BASIS, dict(space = 'BDM1', edges = MESH.Boundary.TrigEdges, order = 2), pex)
p_n_BDM1+= pde.projections.assem_HDIV_b(MESH, BASIS, dict(space = 'BDM1', edges = MESH.Boundary.QuadEdges, order = 0), pex)



Z = sps.coo_matrix((C.shape[0],C.shape[0]), dtype = np.float64)
A = sps.vstack((sps.hstack((Mh,-C.T)),
                sps.hstack(( C,   Z))))


b = np.r_[-p_n_BDM1,
           M_divuex_P0_Q0_new]


res = sps.linalg.spsolve(A,b)

uh = res[:2*MESH.NoEdges]
ph = res[2*MESH.NoEdges:]

uhx_l = sps.linalg.spsolve(M_P1d_Q1d,Mx_P1d*uh)
uhy_l = sps.linalg.spsolve(M_P1d_Q1d,My_P1d*uh)

# import matplotlib
# matplotlib.pyplot.spy(M_P1d_Q1d,markersize = 0.1)



tt = time.time()
cholMh = cholesky(Mh)
for k in range(100):
    b = np.random.rand(Mh.shape[0])
    rs = cholMh.solve_A(b)
print(time.time() - tt)

# fig1 = MESH.pdemesh(info = 1,border = 0.6)
# fig1.show()


fig2 = MESH.pdesurf_hybrid(dict(trig='P1d',quad='Q1d'),uhx_l)
fig2.show()

# TODO : pde.mypcg(A,b,tol)





