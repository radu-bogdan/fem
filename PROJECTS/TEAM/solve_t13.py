from t13geo import *

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
from sksparse.cholmod import cholesky as chol

order = 1

MESH = pde.mesh3(p,e,f,t,regions_3d_np,regions_2d_np)

phi_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = order)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = order)
D = pde.int.assemble3(MESH, order = order)

M = phi_H1 @ D @ phi_H1.T

K = dphix_H1 @ D @ dphix_H1.T +\
    dphiy_H1 @ D @ dphiy_H1.T +\
    dphiz_H1 @ D @ dphiz_H1.T

R0, RSS = pde.h1.assembleR3(MESH, space = 'P1', faces = 'ambient_face')

Kn = RSS @ K @ RSS.T

coeff = lambda x,y,z : 1+0*x*y*z+0*x*y*z
J = pde.int.evaluate3(MESH, order = order, coeff = coeff, regions = 'mid_steel').diagonal()

r = J @ D @ phi_H1.T

# # solve:
u = RSS.T@(chol(Kn).solve_A(RSS@r))
# u = chol(M).solve_A(r)

u2 = coeff(p[:,0],p[:,1],p[:,2])


MESH.pdesurf(u, faces = 'l_steel_face,r_steel_face,mid_steel_face,coil_face')

