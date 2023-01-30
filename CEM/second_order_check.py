import sys
sys.path.insert(0,'..') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import geometries

import plotly.io as pio
pio.renderers.default = 'browser'
from matplotlib.pyplot import spy


np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = np.inf)
np.set_printoptions(precision = 8)

gmsh.initialize()
gmsh.model.add("Capacitor plates")
geometries.unitSquare()
gmsh.option.setNumber("Mesh.Algorithm", 2)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
# gmsh.fltk.run()

p,e,t,q = pde.petq_generate()

MESH = pde.mesh(p,e,t,q)
MESH.makeFemLists()

BASIS = pde.basis()
LISTS = pde.lists(MESH)

f = lambda x,y : (np.pi**2+np.pi**2)*np.sin(np.pi*x)*np.sin(np.pi*y)
g = lambda x,y : 0*x

Kx,Ky = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 0)
D0 = pde.int.assemble(MESH, order = 0)

MB = pde.h1.assemble(MESH, space = 'P1', matrix = 'M', order = 2)
D2 = pde.int.assemble(MESH, order = 2)


Kxx = Kx@D0@Kx.T; Kyy = Ky@D0@Ky.T
M = MB@D2@MB.T

Mb = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = Kxx.shape, order = 2)
Db0 = pde.int.assembleB(MESH, order = 2)
B_full = Mb@Db0@Mb.T


ff = pde.int.evaluate(MESH, coeff = f, order = 2)
M_f = MB@D2@ ff.diagonal()

D_g = pde.int.evaluateB(MESH, order = 2, coeff = g)

B_g = Mb@Db0@D_g.diagonal()


gamma = 10**10

lam = 1
A = Kxx + Kyy -lam*M + gamma*B_full
b = gamma*B_g + M_f

sigma = 3
# x = sps.linalg.eigs(Kxx+Kyy-sigma*M+gamma*B_full,M = M, sigma = sigma)
x = sps.linalg.eigs(Kxx+Kyy,M = M, sigma = sigma)
phi = np.real(x[1][:,0])

# tm = time.time()
# phi = sps.linalg.spsolve(A,b)
# elapsed = time.time()-tm
# print('Solving took ' + str(elapsed)[0:5] + ' seconds.')

fig = MESH.pdesurf_hybrid(dict(trig = 'P1', controls = 1), phi)
fig.show()

