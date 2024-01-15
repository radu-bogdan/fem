
from ngsolve.webgui import Draw
from netgen.webgui import Draw as DrawGeo

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import cholesky as chol

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
import time

import importlib
importlib.reload(pde)

from magnGeo import *

###########################################################################
tm = time.monotonic()
MESH = pde.mesh3.netgen(geoOCCmesh)
print('MESH.stuff ... ',time.monotonic()-tm)
###########################################################################

order = 1
D = pde.int.assemble3(MESH, order = order)
DB = pde.int.assembleB3(MESH, order = order)
unit_coil = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : 1+0*x, regions = 'coil')
face_in  = pde.int.evaluateB3(MESH, order = order, coeff = lambda x,y,z : 1/crosssection+0*x, faces = 'in').diagonal()

###########################################################################

tm = time.monotonic()

phi_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = order)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = order)
phiB_H1 = pde.h1.assembleB3(MESH, space = 'P1', matrix = 'M', shape = phi_H1.shape, order = order)

R0, RSS = pde.h1.assembleR3(MESH, space = 'P1', faces = 'out')

r = face_in @ DB @ phiB_H1.T

M = phi_H1 @ D @ unit_coil @ phi_H1.T

K = dphix_H1 @ D @ unit_coil @ dphix_H1.T +\
    dphiy_H1 @ D @ unit_coil @ dphiy_H1.T +\
    dphiz_H1 @ D @ unit_coil @ dphiz_H1.T


K = RSS @ K @ RSS.T
RZ = pde.tools.removeZeros(K)
K = RZ @ K @ RZ.T

M = RSS @ M @ RSS.T
M = RZ @ M @ RZ.T

r = RZ @ RSS @ r

sigma = 58.7e6
x = chol(sigma*K).solve_A(r)
x = RSS.T @ RZ.T @ x
print('My code took ... ',time.monotonic()-tm)

###########################################################################


phix_Hdiv, phiy_Hdiv, phiz_Hdiv = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'M', order = order)
divphi_Hdiv = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'K', order = order)
phi_L2 = pde.l2.assemble3(MESH, space = 'P0', matrix = 'M', order = order)

M_Hdiv = phix_Hdiv @ D @ unit_coil @ phix_Hdiv.T +\
         phiy_Hdiv @ D @ unit_coil @ phiy_Hdiv.T +\
         phiz_Hdiv @ D @ unit_coil @ phiz_Hdiv.T

C_Hdiv_L2 = divphi_Hdiv @ D @ unit_coil @ phi_L2.T



dsa = pde.hdiv.assembleB3(MESH, space = 'RT0', matrix = 'M', shape = phix_Hdiv.shape, order = 1)

A = bmat([[M_Hdiv,-C_Hdiv_L2],
          [C_Hdiv_L2.T, None]])

import vtklib

grid = vtklib.createVTK(MESH)
vtklib.add_H1_Scalar(grid, x, 'lel')
vtklib.writeVTK(grid, 'das.vtu')
