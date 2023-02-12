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

gmsh.initialize()
gmsh.model.add("Capacitor plates")
geometries.geometryP2()
gmsh.option.setNumber("Mesh.Algorithm", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.MeshSizeMax",0.05)

# gmsh.fltk.run()
# quit()

p,e,t,q = pde.petq_generate()
gmsh.clear()
gmsh.finalize()

MESH = pde.mesh(p,e,t,q)
MESH.makeFemLists(space = 'P1')
MESH.makeFemLists(space = 'P0')

# BASIS = pde.basis()
# LISTS = pde.lists(MESH)

f1 = lambda x,y : -1+0*x
f2 = lambda x,y :  1+0*x

nu1 = lambda x,y : 1/1000 + 0*x +0*y
nu2 = lambda x,y : 1 + 0*x +0*y

BKx,BKy = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 0)
D2 = pde.int.assemble(MESH, order = 2)
D0 = pde.int.assemble(MESH, order = 0)

Co1 = pde.int.evaluate(MESH, order = 0, coeff = nu1, regions = np.r_[2,3])
Co2 = pde.int.evaluate(MESH, order = 0, coeff = nu2, regions = np.r_[1,4,5,6,7,8])

Kxx = BKx@D0@(Co1+Co2)@BKx.T; Kyy = BKy@D0@(Co1+Co2)@BKy.T

BM = pde.h1.assemble(MESH, space = 'P1', matrix = 'M', order = 2)
M = BM@D2@BM.T

D = pde.l2.assemble(MESH, space = 'P0', matrix = 'M')

CoF1 = pde.int.evaluate(MESH, order = 2, coeff = f1, regions = np.r_[7])
CoF2 = pde.int.evaluate(MESH, order = 2, coeff = f2, regions = np.r_[8])
M_f = BM@D2@(CoF1.diagonal()+CoF2.diagonal())

Mb = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = Kxx.shape)
Db2 = pde.int.assembleB(MESH, order = 2)

B = Mb@Db2@Mb.T

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

ux = BKx.T@u
uy = BKy.T@u

u_P0 = 1/3*(u[MESH.t[:,0]]+u[MESH.t[:,1]]+u[MESH.t[:,2]])
eu = nu_vek*1/2*ux**2+uy**2-j_vek*u_P0
Eu = 1/2*u@(Kxx+Kyy)@u
print(Eu)


fig = MESH.pdesurf_hybrid(dict(trig = 'P1',quad = 'Q1', controls = 1), u)
fig.show()

fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0', controls = 1), eu)
fig.show()