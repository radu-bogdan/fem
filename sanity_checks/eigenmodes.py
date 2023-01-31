import sys
sys.path.insert(0,'..') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

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
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.02)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.02)

f = lambda x,y : (np.pi**2+np.pi**2)*np.sin(np.pi*x)*np.sin(np.pi*y)
g = lambda x,y : 0*x
u = lambda x,y : np.sin(np.pi*x)*np.sin(np.pi*y)

p,e,t,q = pde.petq_generate()
MESH = pde.mesh(p,e,t,q)
MESH.makeFemLists()
    
tm = time.time()

# Mass & Stifness    
Kx,Ky = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 0)
D0 = pde.int.assemble(MESH, order = 0)

MB = pde.h1.assemble(MESH, space = 'P1', matrix = 'M', order = 2)
D2 = pde.int.assemble(MESH, order = 2)

Kxx = Kx@D0@Kx.T; Kyy = Ky@D0@Ky.T; M = MB@D2@MB.T

ff = pde.int.evaluate(MESH, coeff = f, order = 2)
M_f = MB@D2@ ff.diagonal()

# Boundary stuff
Mb = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = Kxx.shape, order = 2)
Db0 = pde.int.assembleB(MESH, order = 2)
B_full = Mb@Db0@Mb.T

D_g = pde.int.evaluateB(MESH, order = 2, coeff = g)
B_g = Mb@Db0@ D_g.diagonal()

gamma = 10**10

A = Kxx + Kyy + gamma*B_full
b = gamma*B_g + M_f

u_ex = u(MESH.p[:,0],MESH.p[:,1])
elapsed = time.time()-tm; print('Assembling stuff took ' + str(elapsed)[0:5] + ' seconds.')

u = MB@D2@ pde.int.evaluate(MESH, coeff = f, order = 2).diagonal()

sigma = 1
# x = sps.linalg.eigs(Kxx+Kyy-sigma*M+gamma*B_full,M = M, sigma = sigma)
x = sps.linalg.eigsh(Kxx + Kyy+ 10**10*B_full, M = 2*np.pi**2*M, sigma = 1, k = 100)
phi = np.real(x[1][:,5])

print(x[0])

# tm = time.time()
# uh = sps.linalg.spsolve(A,b)
# elapsed = time.time()-tm
# print('Solving took ' + str(elapsed)[0:5] + ' seconds.')


fig = MESH.pdesurf_hybrid(dict(trig = 'P1', controls = 1), phi)
fig.show()

