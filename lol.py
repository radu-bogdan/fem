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



MESH = pde.initmesh(p,e,t,q)
BASIS = pde.basis()
LISTS = pde.lists(MESH)

MAT = {}
# MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = 'RT0-RT0')
MAT = MAT | pde.assemble.hdiv(MESH,BASIS,LISTS,space = 'BDM1-BDM1')
# MAT = MAT | pde.assemble.h1(MESH,BASIS,LISTS,space = 'P1-Q1')
MAT = MAT | pde.assemble.h1(MESH,BASIS,LISTS,space = 'P1d-Q1d')
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





# # import plotly.figure_factory as ff

# # from scipy.spatial import Delaunay
# import plotly.graph_objects as go

# # # Make data for plot
# # u = np.linspace(0, 2*np.pi, 20)
# # v = np.linspace(0, np.pi, 20)
# # u,v = np.meshgrid(u,v)
# # u = u.flatten()
# # v = v.flatten()
# # x = np.sin(v)*np.cos(u)
# # y = np.sin(v)*np.sin(u)
# # z = np.cos(v)

# # points2D = np.vstack([u,v]).T
# # tri = Delaunay(points2D)
# # simplices = tri.simplices

# u = pex_l

# zz_trig = u[0:3*MESH.nt].reshape(MESH.nt,3)
# zz_quad_orig = u[3*MESH.nt:].reshape(MESH.nq,4)

# xx_trig = np.c_[p[t[:,0],0],p[t[:,1],0],p[t[:,2],0]]
# yy_trig = np.c_[p[t[:,0],1],p[t[:,1],1],p[t[:,2],1]]

# xx_quad_1 = np.c_[p[q[:,0],0],p[q[:,1],0],p[q[:,2],0],]
# yy_quad_1 = np.c_[p[q[:,0],1],p[q[:,1],1],p[q[:,2],1]]
# zz_quad_1 = zz_quad_orig[:,0:3]

# xx_quad_2 = np.c_[p[q[:,0],0],p[q[:,2],0],p[q[:,3],0]]
# yy_quad_2 = np.c_[p[q[:,0],1],p[q[:,2],1],p[q[:,3],1]]
# zz_quad_2 = np.c_[zz_quad_orig[:,0],zz_quad_orig[:,2],zz_quad_orig[:,3]]

# xx_quad = np.r_[xx_quad_1,xx_quad_2]
# yy_quad = np.r_[yy_quad_1,yy_quad_2]
# zz_quad = np.r_[zz_quad_1,zz_quad_2]


# fig2 = go.Figure()


# ii, jj, kk = np.r_[0:3*MESH.nt].reshape((MESH.nt, 3)).T
# fig2.add_trace(go.Mesh3d(
#     x = xx_trig.flatten(), y = yy_trig.flatten(), z = zz_trig.flatten(),
#     i = ii, j = jj, k = kk, intensity = zz_trig.flatten(), 
#     colorscale = 'Jet',
#     cmin = np.min(u),
#     cmax = np.max(u),
#     lighting = dict(ambient = 1)
# ))


# ii, jj, kk = np.r_[0:3*MESH.nq+3*MESH.nq].reshape((MESH.nq+MESH.nq, 3)).T
# fig2.add_trace(go.Mesh3d(
#     x = xx_quad.flatten(), y = yy_quad.flatten(), z = zz_quad.flatten(),
#     i = ii, j = jj, k = kk, intensity = zz_quad.flatten(), 
#     colorscale = 'Jet',
#     cmin = np.min(u),
#     cmax = np.max(u),
#     lighting = dict(ambient = 1)
# ))

# xxx_trig = np.c_[xx_trig,xx_trig[:,0],np.nan*xx_trig[:,0]]
# yyy_trig = np.c_[yy_trig,yy_trig[:,0],np.nan*yy_trig[:,0]]
# zzz_trig = np.c_[zz_trig,zz_trig[:,0],np.nan*zz_trig[:,0]]


# xxx_quad = np.c_[p[q[:,0],0],p[q[:,1],0],p[q[:,2],0],p[q[:,3],0],p[q[:,0],0],np.nan*p[q[:,0],0]]
# yyy_quad = np.c_[p[q[:,0],1],p[q[:,1],1],p[q[:,2],1],p[q[:,3],1],p[q[:,0],1],np.nan*p[q[:,0],1]]
# zzz_quad = np.c_[zz_quad_orig,zz_quad_orig[:,0],np.nan*zz_quad_orig[:,0]]


# fig2.add_trace(go.Scatter3d(mode='lines',
#                             x=xxx_trig.flatten(),
#                             y=yyy_trig.flatten(),
#                             z=zzz_trig.flatten(),
#                             line=go.scatter3d.Line(color='black', width=1.5),
#                             showlegend=False))

# fig2.add_trace(go.Scatter3d(mode='lines',
#                             x=xxx_quad.flatten(),
#                             y=yyy_quad.flatten(),
#                             z=zzz_quad.flatten(),
#                             line=go.scatter3d.Line(color='black', width=1.5),
#                             showlegend=False))



    
# camera = dict(eye=dict(x=0, y=-1e-5, z=1.5))
# # camera = dict(up=dict(x=0, y=1, z=0),eye=dict(x=0, y=0, z=1),)
# ratio = (max(p[:,0])-min(p[:,0]))/(max(p[:,1])-min(p[:,1]))
# fig2.update_layout(scene=dict(aspectratio=dict(x=ratio, y=1, z=1)),scene_camera=camera)
# fig2.layout.scene.camera.projection.type = "orthographic"

# fig2.show()






# print('here')











