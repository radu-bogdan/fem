from netgen.csg import *
from netgen.geom2d import CSG2d, Circle, Rectangle
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
from sksparse.cholmod import cholesky as chol

# 1) Creation of the Solids

Cirle1 =  Circle( center=(0,0), radius=0.1, bc = "in" )
Cirle2 = Circle( center=(0,0), radius=0.2, bc = "out" )
In = Cirle2 - Cirle1

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde



# 2) Creation of the Geometry

Omega2D = CSG2d()
Omega2D.Add(In)
####################################

mesh2D = Mesh(Omega2D.GenerateMesh(maxh=0.01))
MESH = pde.mesh.netgen(mesh2D.ngmesh)

##########################################################################################
# Assembling stuff
##########################################################################################

dphix_H1, dphiy_H1 = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 1)
D = pde.int.assemble(MESH, order = 1)

K = dphix_H1 @ D @ dphix_H1.T + dphiy_H1 @ D @ dphiy_H1.T

Z, RS = pde.h1.assembleR(MESH, space = 'P1', edges = 'in,out')
Ri, Z = pde.h1.assembleR(MESH, space = 'P1', edges = 'in')

KN = RS @ K @ RS.T

x = RS.T @ chol(KN).solve_A(-RS @ K @ Ri.T@(1+np.zeros(Ri.shape[0]))) + Ri.T@(1+np.zeros(Ri.shape[0]))
MESH.pdesurf(x)

##########################################################################################