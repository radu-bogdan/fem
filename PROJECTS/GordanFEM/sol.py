#!/usr/bin/python --relpath_append ../
# @profile
# def kek():

import sys
sys.path.insert(0,'../../') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
from sksparse.cholmod import cholesky
import meshio

from matplotlib.pyplot import spy
import plotly.io as pio
pio.renderers.default = 'browser'
# pio.renderers.default = 'svg'

from scipy.constants import mu_0


np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = np.inf)
np.set_printoptions(precision = 8)

p,e,t,q = pde.petq_from_gmsh(filename = '2_conductors_x.geo',hmax = 0.2)

# mesh = meshio.read("2_conductors_x_Gordan.msh")


tm = time.time()
# p,e,t,q = pde.petq_generate()

MESH = pde.mesh(p,e,t,q)

MESH.pdemesh2()

print('Generating mesh and refining took {:4.8f} seconds.'.format(time.time()-tm))
# MESH.makeRest()

# BASIS = pde.basis()
# LISTS = pde.lists(MESH)

f1 = lambda x,y : 0*x+10**3/(0.1**2*np.pi)
f2 = lambda x,y : 0*x-10**3/(0.1**2*np.pi)
f3 = lambda x,y : 0*x

###############################################################################
tm = time.time()

Kx,Ky = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 0)
D0 = pde.int.assemble(MESH, order = 0)

Kxx = Kx@D0@Kx.T; Kyy = Ky@D0@Ky.T

walls = np.r_[1]

Mb = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = Kxx.shape, order = 2)
Db0 = pde.int.assembleB(MESH, order = 2)


# D_left_block = pde.int.evaluateB(MESH, order = 2, edges = left_block)
# D_right_block = pde.int.evaluateB(MESH, order = 2, edges = right_block)
D_walls = pde.int.evaluateB(MESH, order = 2, edges = walls)

B_full = Mb@Db0@Mb.T
# B_left_block = Mb@Db0@D_left_block@Mb.T
# B_right_block = Mb@Db0@D_right_block@Mb.T
B_walls = Mb@Db0@D_walls@Mb.T

# D_g1 = pde.int.evaluateB(MESH, order = 2, edges = np.r_[1], coeff = f1)
D_g2 = pde.int.evaluateB(MESH, order = 2, edges = np.r_[2], coeff = f2)
# D_g = pde.int.evaluateB(MESH, order = 2, edges = np.r_[2], coeff = f3)

# f1 = pde.int.evaluate(MESH, order = 2, regions = np.r_[1], coeff = f1)


vb = D_walls@D_g2.diagonal()
B_g = Mb@Db0@vb


# R = makeR(D_left_block@D_g1.diagonal() + D_right_block@D_g2.diagonal())

gamma = 10**8

# R = sps.eye(Kxx.shape[0],format = 'csc')
# R

A = Kxx + Kyy + gamma*B_walls
b = 0*B_g

print('Assembling took {:4.8f} seconds.'.format(time.time()-tm))
###############################################################################


tm = time.time()

cholA = cholesky(A)
phi = cholA(b)
# phi = sps.linalg.spsolve(A,b)

oneM = np.ones(Kxx.shape[0])
# Q = gamma*(oneM@B_left_block@(phi+1)) #+ oneM@B_right_block@(phi-1)
# Q = gamma*(oneM@B_right_block@(phi-1))

# print(str(Q/2), ' soll: ', str(l/d))

elapsed = time.time()-tm
print('Solving took {:4.8f} seconds.'.format(elapsed))

fig = MESH.pdesurf_hybrid(dict(trig = 'P1', controls = 1), phi)
fig.show()

# kek()