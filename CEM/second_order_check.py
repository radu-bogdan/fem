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
gmsh.option.setNumber("Mesh.MeshSizeMax", 2)
# gmsh.fltk.run()
p,e,t,q = pde.petq_generate()

MESH = pde.initmesh(p,e,t,q)

# stop()

# TODO:  MESH = pde.refinemesh(p,e,t,q)
BASIS = pde.basis()
LISTS = pde.lists(MESH)

f = lambda x,y : (np.pi**2*np.pi**2)*np.sin(np.pi*x)*np.sin(np.pi*y)
g = lambda x,y : 0*x

# TODO : iwas stimmt net wenn ma quads hat
Kxx,Kyy,Kxy,Kyx = pde.assemble.h1(MESH,BASIS,LISTS,dict(space = 'P2', matrix = 'K'))
M = pde.assemble.h1(MESH,BASIS,LISTS,dict(space = 'P2', matrix = 'M'))

sizeM = M.shape[0]

B_full = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P2', size = sizeM))
M_f = pde.projections.assemH1(MESH, BASIS, LISTS, dict(trig = 'P2'), f)

B_g  = pde.projections.assem_H1_b(MESH, BASIS, LISTS, dict(space = 'P2', order = 2, size = sizeM), g)

gamma = 10**8

A = Kxx + Kyy + gamma*B_full
b = gamma*B_g + M_f



tm = time.time()
phi = sps.linalg.spsolve(A,b)
elapsed = time.time()-tm
print('Solving took ' + str(elapsed)[0:5] + ' seconds.')

fig = MESH.pdesurf_hybrid(dict(trig = 'P1', controls = 1), phi[0:MESH.np])
fig.show()

