# from geo_and_mesh import *
from ngsolve import *
from netgen.csg import *
from math import pi
from scipy import interpolate
import ngsolve as ng
import numpy as np
import time
from ngsolve.solvers import *
from netgen.geom2d import SplineGeometry
from ngsolve.internal import visoptions
from ngsolve.internal import *
from netgen.occ import *
from netgen.meshing import IdentificationType
from collections import defaultdict
import scipy.sparse as sp
import scipy
import scipy.sparse.linalg as scspla

separated_mesh = True


exec(open("geo_and_mesh.py").read())

import sys
sys.path.insert(0,'../../../') # adds parent directory
# import pde
import scipy.linalg
# import netgen.gui


fes = H1(mesh=mesh, order=1, dirichlet='stator_outer')
u,v = fes.TnT()
gfu = GridFunction(fes)

a = BilinearForm(fes)
a += (grad(u)[0] * grad(v)[0] + grad(u)[1] * grad(v)[1] ) * dx
a.Assemble()

rowsA, colsA, valsA = a.mat.COO()
K = sp.csr_matrix((valsA,(rowsA,colsA)))

identications = np.array(mesh.ngmesh.GetIdentifications())
points_left = identications[:,1]
points_right = identications[:,0]

idces_airgap_array, idces_corners_array, idces_stator_outer_array = identifyPointsAirgap()
idces_stator_outer_array = np.array(idces_stator_outer_array)

airgap_top = idces_airgap_array[:,1]
airgap_bottom = idces_airgap_array[:,0]


airgap_left_top = idces_corners_array[1,1]
airgap_left_bottom = idces_corners_array[1,0]

airgap_right_top = idces_corners_array[0,1]
airgap_right_bottom = idces_corners_array[0,0]



corner_outer_right = np.intersect1d(idces_stator_outer_array, points_right)
corner_outer_left  = np.intersect1d(idces_stator_outer_array, points_left)
corner_tip = np.intersect1d(points_left, points_right)

allBoundary = np.unique(np.concatenate([identications.ravel(),idces_airgap_array.ravel(),idces_corners_array.ravel(),idces_stator_outer_array.ravel()]))
allInterior = np.setdiff1d(np.r_[:mesh.nv], allBoundary)

points_left_nocorners = np.setdiff1d(points_left, np.r_[corner_tip, corner_outer_left, airgap_left_top, airgap_left_bottom])
points_right_nocorners = np.setdiff1d(points_right, np.r_[corner_tip, corner_outer_right, airgap_right_top, airgap_right_bottom])

# points_left_nocorners = np.setdiff1d(points_left, np.r_[corner_tip,corner_outer_left,airgap_left_bottom])
# points_right_nocorners = np.setdiff1d(points_right, np.r_[corner_tip,corner_outer_right,airgap_right_bottom])

# points_left_nocorners = np.setdiff1d(points_left, np.r_[corner_tip,corner_outer_left,airgap_left_bottom])
# points_right_nocorners = np.setdiff1d(points_right, np.r_[corner_tip,corner_outer_right,airgap_right_bottom])

R_int = sp.eye(mesh.nv, format = 'csc')
R_int = R_int[allInterior-1,:]

R_airbottom = sp.eye(mesh.nv, format = 'csc')
R_airbottom = R_airbottom[airgap_bottom-1,:]

R_airtop = sp.eye(mesh.nv, format = 'csc')
R_airtop = R_airtop[airgap_top-1,:]


R_left = sp.eye(mesh.nv, format = 'csc')
R_left = R_left[points_left_nocorners-1,:]

R_right = sp.eye(mesh.nv, format = 'csc')
R_right = R_right[points_right_nocorners-1,:]


from scipy.sparse import bmat
RS =  bmat([[R_int], [R_left-R_right], [R_airbottom + R_airtop]])
# RS =  bmat([[R_int], [R_airbottom + R_airtop]])


sv = np.zeros(mesh.nv)
sv[airgap_left_top-1] = 1
sv[airgap_left_bottom-1] = 1
svR1 = RS@sv
svR1 = np.argwhere(svR1)[0]

sv = np.zeros(mesh.nv)
sv[airgap_right_top-1] = 1
sv[airgap_right_bottom-1] = 1
svR2 = RS@sv
svR2 = np.argwhere(svR2)[0]


RP_int = sp.eye(RS.shape[0], format = 'csc')
RP_int= RP_int[np.setdiff1d(np.r_[:RS.shape[0]],np.r_[svR1,svR2]),:]

RP_left = sp.eye(RS.shape[0], format = 'csc')
RP_left = RP_left[svR2,:]

RP_right = sp.eye(RS.shape[0], format = 'csc')
RP_right = RP_right[svR1,:]

RS2 =  bmat([[RP_int], [RP_left-RP_right]])




K_new = RS2 @ RS @ K @ RS.T @ RS2.T 

rhs = LinearForm(fes)
rhs += CoefficientFunction(1000) * v * dx(definedon = mesh.Materials("magnet1"))
rhs.Assemble()

rhs_np = RS2 @ RS @ rhs.vec

solvec = RS.T @ RS2.T @ sp.linalg.spsolve(K_new, rhs_np) 

gfu.vec.data = solvec 

Draw(gfu,mesh,"whatever")




