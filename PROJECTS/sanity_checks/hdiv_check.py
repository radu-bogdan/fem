import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
from sksparse.cholmod import cholesky
import time
import gmsh
import geometries
# from scikits import umfpack

import plotly.io as pio
pio.renderers.default = 'browser'
np.set_printoptions(precision = 8)

# np.set_printoptions(linewidth = 150)
# p,e,t,q = pde.petq_from_gmsh(filename = 'unit_square.geo',hmax = 1*1/np.sqrt(2)**3)


gmsh.initialize()
gmsh.model.add("Capacitor plates")
# geometries.unitSquare()
gmsh.open('../../unused/mesh_new.geo')
gmsh.option.setNumber("Mesh.Algorithm", 2)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)
p,e,t,q = pde.petq_generate()
gmsh.finalize()

# gmsh.open(filename = 'mesh_new.geo')




# p,e,t,q = pde.petq_from_gmsh(filename = 'mesh_new.geo', hmax = 1/np.sqrt(2)**2)
MESH = pde.mesh(p,e,t,q)
# MESH.refinemesh()
# MESH.refinemesh()
MESH.refinemesh()
MESH.refinemesh()

# BASIS = pde.basis()
# LISTS = pde.lists(MESH)

pex = lambda x,y : (1-x)**4+(1-y)**3*(1-x) + np.sin(1-y)*np.cos(1-x)
u1ex = lambda x,y : -np.sin(1-x)*np.sin(1-y)-4*(x-1)**3-(y-1)**3
u2ex = lambda x,y : -3*(x-1)*(y-1)**2 + np.cos(1-x)*np.cos(1-y)
divuex = lambda x,y : -6*(x-1)*(2*x+y-3)+2*np.cos(1-x)*np.sin(1-y)

# c1 = 2;
# c2 = 1;

# pex = lambda x,y : c1*x + c2*y
# u1ex = lambda x,y : -c1 + 0*x
# u2ex = lambda x,y : -c2 + 0*x
# divuex = lambda x,y : 0 + 0*x

# fig = MESH.pdemesh(info=0,border=0.6)
# fig.show()



qMhx,qMhy = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'M', order = 1)
qMx,qMy = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'M', order = 2)
qMb = pde.hdiv.assembleB(MESH, space = 'BDM1', matrix = 'M', order = 2, shape = 2*MESH.NoEdges)

qK = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'K', order = 2)
qD = pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = 1)

D2 = pde.int.assemble(MESH, order = 2)
D1 = pde.int.assemble(MESH, order = 1)
D0 = pde.int.assemble(MESH, order = 0)

D2b = pde.int.assembleB(MESH, order = 2)
D1b = pde.int.assembleB(MESH, order = 1)
D0b = pde.int.assembleB(MESH, order = 0)

M = qMx@D2@qMx.T + qMy@D2@qMy.T
Mh = qMhx@D1@qMhx.T + qMhy@D1@qMhy.T
K = qK@D2@qK.T
C = qD@D1@qK.T
D = qD@D1@qD.T


eval_divuex = pde.int.evaluate(MESH, order = 2, coeff = divuex).diagonal()
M_divuex = qD@D1@eval_divuex

eval_pex_B = pde.int.evaluateB(MESH, order = 2, coeff = pex).diagonal()
p_n_BDM1 = qMb@D2b@eval_pex_B

# BASIS = pde.basis()
# MESH.makeRest()
# p_n_BDM1b  = pde.projections.assem_HDIV_b(MESH, BASIS, dict(space = 'BDM1', order = 2), pex)


A1 = sps.vstack((sps.hstack((Mh,-C.T)),
                 sps.hstack(( C, 0*D))))

b1 = np.r_[-p_n_BDM1,
            M_divuex]

res1 = sps.linalg.spsolve(A1,b1)

ph1 = res1[2*MESH.NoEdges:]
t = time.time()
fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1), ph1)
print("Took",str(time.time()-t)[:5], "seconds.")
# fig.show()

eval_pex = pde.int.evaluate(MESH, order = 0, coeff = pex).diagonal()
fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1), eval_pex)
print("Took",str(time.time()-t)[:5], "seconds.")
# fig.show()


from matplotlib.pyplot import spy
cholM = cholesky(Mh)
spy(cholM.L(), markersize = 1)