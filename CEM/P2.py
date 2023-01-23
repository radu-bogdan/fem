#!/usr/bin/python --relpath_append ../

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

np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = np.inf)
np.set_printoptions(precision = 8)

# p,e,t,q = pde.petq_from_gmsh(filename = 'unit_square.geo',hmax = 0.3)

# gmsh.initialize()


gmsh.initialize()
gmsh.model.add("Capacitor plates")
geometries.geometryP2()
gmsh.option.setNumber("Mesh.Algorithm", 2)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.MeshSizeMax",0.1)
# gmsh.fltk.run()

# quit()

p,e,t,q = pde.petq_generate()
gmsh.clear()
gmsh.finalize()

MESH = pde.initmesh(p,e,t,q)
# fig = MESH.pdemesh()
# fig.show()

# TODO:  MESH = pde.refinemesh(p,e,t,q)
BASIS = pde.basis()
LISTS = pde.lists(MESH)

f1 = lambda x,y : -1+0*x
f2 = lambda x,y :  1+0*x

nu1 = lambda x,y : 1/1000 + 0*x +0*y
nu2 = lambda x,y : 1 + 0*x +0*y

# TODO : iwas stimmt net wenn ma quads hat

Kxx1,Kyy1,Kxy1,Kyx1 = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'K', coeff = nu1, regions = np.r_[2,3]))
Kxx2,Kyy2,Kxy2,Kyx2 = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'K', coeff = nu2, regions = np.r_[1,4,5,6,7,8]))
Kxx = Kxx1 + Kxx2; Kyy = Kyy1 + Kyy2

M = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'M'))

Cx,Cy = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'C'))

MAT = pde.assemble.hdiv(MESH, BASIS, LISTS, space = 'BDM1-BDM1'); D = MAT['BDM1-BDM1']['D'];

B = pde.assemble.h1b(MESH,BASIS,LISTS, dict(space = 'P1', edges = np.r_[1,2,3,4], size = Kxx.shape[0]))

M_f_1 = pde.projections.assemH1(MESH, BASIS, LISTS, dict(trig = 'P1', regions = np.r_[7]), f1)
M_f_2 = pde.projections.assemH1(MESH, BASIS, LISTS, dict(trig = 'P1', regions = np.r_[8]), f2)

M_f = M_f_1 + M_f_2

A = Kxx + Kyy + 10**10*B
b = M_f

tm = time.time()
u = sps.linalg.spsolve(A,b)
elapsed = time.time()-tm
print('Solving took ' + str(elapsed)[0:5] + ' seconds.')


nu_vek  = pde.projections.evaluateP0_trig(MESH, dict(regions = np.r_[2,3]), nu1)
nu_vek += pde.projections.evaluateP0_trig(MESH, dict(regions = np.r_[1,4,5,6,7,8]), nu2)

j_vek  = pde.projections.evaluateP0_trig(MESH, dict(regions = np.r_[7]), f1)
j_vek += pde.projections.evaluateP0_trig(MESH, dict(regions = np.r_[8]), f2)

ux = sps.linalg.spsolve(D,Cx*u)
uy = sps.linalg.spsolve(D,Cy*u)

u_P0 = 1/3*(u[MESH.t[:,0]]+u[MESH.t[:,1]]+u[MESH.t[:,2]])
eu = nu_vek*1/2*ux**2+uy**2-j_vek*u_P0
Eu = 1/2*u@(Kxx+Kyy)@u
print(Eu)


fig = MESH.pdesurf_hybrid(dict(trig = 'P1',quad = 'Q1', controls = 1), u)
fig.show()

fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0', controls = 1), eu)
fig.show()
