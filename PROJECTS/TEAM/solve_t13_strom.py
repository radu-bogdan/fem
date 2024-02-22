from t13geo import *

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
from sksparse.cholmod import cholesky as chol
import numpy as np
import time
import scipy.sparse as sp

MESH = pde.mesh3.netgen(geoOCCmesh)

##############################################################################

face_index = pde.tools.getIndices(MESH.regions_2d, 'coil_cut_1')
faces = MESH.f[MESH.BoundaryFaces_Region == face_index,:3]
new_faces = faces.copy()

points_to_duplicate = np.unique(faces.ravel())
new_points = np.arange(MESH.np, MESH.np+points_to_duplicate.size)

actual_points = MESH.p[points_to_duplicate,:]

t_new = MESH.t[:,:4].copy()
f_new = MESH.f[:,:3].copy()
p_new = MESH.p.copy()

for i,pnt in enumerate(points_to_duplicate):

    # append point to list
    p_new = np.vstack([p_new,p_new[pnt,:]])

    # finding tets coordinates containing the ith point to duplicate
    tets_containing_points = np.argwhere(t_new[:,:4]==pnt)[:,0]

    for _,j in enumerate(tets_containing_points):
        #check if tet is left
        if MESH.mp_tet[j,0]<0:
            t_new[j,t_new[j,:]==pnt] = MESH.np + i
            
    # finding faces containing the points
    faces_containing_points = np.argwhere(f_new[:,:3]==pnt)[:,0]
    for _,j in enumerate(faces_containing_points):
        #check if face is left
        if 1/3*(p_new[f_new[j,0],0] + p_new[f_new[j,1],0] + p_new[f_new[j,2],0])<0:
            f_new[j,f_new[j,:]==pnt] = MESH.np + i
            
            
    # print(faces_containing_points)

t_new = np.c_[t_new,MESH.t[:,4]]

for i,j in enumerate(faces.ravel()):
    new_faces.ravel()[i] = new_points[points_to_duplicate==j][0]

new_faces = np.c_[new_faces,np.tile(MESH.f[:,3].max()+1,(new_faces.shape[0],1))]
f_new = (np.r_[np.c_[f_new,MESH.f[:,3]],new_faces]).astype(int)

regions_2d_new = MESH.regions_2d.copy()
regions_2d_new.append('new')

identifications = (np.c_[points_to_duplicate,new_points]).astype(int)
MESH = pde.mesh3(p_new,MESH.e,f_new,t_new,MESH.regions_3d,regions_2d_new,MESH.regions_1d,identifications = identifications)

##############################################################################
sigma = 6*1e7
scaling = 0.004375 # Volts -> about 1000A
##############################################################################

order = 1
D = pde.int.assemble3(MESH, order = order)
DB = pde.int.assembleB3(MESH, order = order)
unit_coil = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : 1+0*x, regions = 'coil')

##############################################################################

tm = time.monotonic()

phi_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = order)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = order)

phiB_H1 = pde.h1.assembleB3(MESH, space = 'P1', matrix = 'phi', shape = phi_H1.shape, order = order)
n_phiB_H1 = pde.h1.assembleB3(MESH, space = 'P1', matrix = 'n*phi', shape = phi_H1.shape, order = order)

R0, RS0 = pde.h1.assembleR3(MESH, space = 'P1', faces = 'new,coil_cut_1')
R1, RS1 = pde.h1.assembleR3(MESH, space = 'P1', faces = 'coil_cut_1')

##############################################################################

K = dphix_H1 @ D @ unit_coil @ dphix_H1.T +\
    dphiy_H1 @ D @ unit_coil @ dphiy_H1.T +\
    dphiz_H1 @ D @ unit_coil @ dphiz_H1.T

r = -RS0 @ K @ R1.T @ (scaling + np.zeros(R1.shape[0]))
K = RS0 @ K @ RS0.T

RZ = pde.tools.removeZeros(K)
K = RZ @ K @ RZ.T

r = RZ @ r

# stop

x = chol(K).solve_A(r)
potential_H1 = RS0.T @ RZ.T @ x + R1.T @ (scaling + np.zeros(R1.shape[0]))
print('My code computing J in L2 took ... ',time.monotonic()-tm)

# stop

##############################################################################
# Hdiv solver
##############################################################################

tm = time.monotonic()

order = 1
phix_Hdiv, phiy_Hdiv, phiz_Hdiv = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'M', order = order)
divphi_Hdiv = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'K', order = order)

phi_L2 = pde.l2.assemble3(MESH, space = 'P0', matrix = 'M', order = order)

D = pde.int.assemble3(MESH, order = order)

M_Hdiv_coil_full = phix_Hdiv @ D @ unit_coil @ phix_Hdiv.T +\
                   phiy_Hdiv @ D @ unit_coil @ phiy_Hdiv.T +\
                   phiz_Hdiv @ D @ unit_coil @ phiz_Hdiv.T

C_Hdiv_L2 = divphi_Hdiv @ D @ unit_coil @ phi_L2.T
R1, RS1 = pde.hdiv.assembleR3(MESH, space = 'RT0', faces = 'coil_face')

M_Hdiv_coil_full = RS1 @ M_Hdiv_coil_full @RS1.T
C_Hdiv_L2 = RS1 @ C_Hdiv_L2

AA = sp.bmat([[M_Hdiv_coil_full, C_Hdiv_L2],
              [C_Hdiv_L2.T, None]])

RZdiv = pde.tools.removeZeros(AA)
AA = RZdiv @ AA @ RZdiv.T

phiB_Hdiv = pde.hdiv.assembleB3(MESH, space = 'RT0', matrix = 'phi', shape = phix_Hdiv.shape, order = order)
unit_coil_B = pde.int.evaluateB3(MESH, order = order, coeff = lambda x,y,z : 1+0*x, faces = 'new').diagonal()
DB = pde.int.assembleB3(MESH, order = order)

rhs = scaling*unit_coil_B @ DB @ phiB_Hdiv.T
# print(rhs[rhs!=0],np.argwhere(rhs!=0))
# rhs[np.argwhere(rhs!=0)[14]]=-0.5

rhs = np.r_[RS1@rhs,np.zeros(MESH.nt)]
# print(rhs[rhs!=0],np.argwhere(rhs!=0))

rhs = RZdiv @ rhs


xx = sp.linalg.spsolve(AA,rhs)

potential_L2 = (RZdiv.T@xx)[-MESH.nt:]
j_hdiv = sigma*RS1.T@(RZdiv.T@xx)[:-MESH.nt]
print('My code computing J in Hdiv (mixed) took ... ',time.monotonic()-tm)

##############################################################################
unit_coil_P0 = pde.int.evaluate3(MESH, order = 0, coeff = lambda x,y,z : 1+0*x, regions = 'coil')

jx_hdiv = (phix_Hdiv.T@j_hdiv)*unit_coil.diagonal()
jy_hdiv = (phiy_Hdiv.T@j_hdiv)*unit_coil.diagonal()
jz_hdiv = (phiz_Hdiv.T@j_hdiv)*unit_coil.diagonal()

phix_Hdiv_P0, phiy_Hdiv_P0, phiz_Hdiv_P0 = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'M', order = 0)
jx_hdiv_P0 = (phix_Hdiv_P0.T@j_hdiv)*unit_coil_P0.diagonal()
jy_hdiv_P0 = (phiy_Hdiv_P0.T@j_hdiv)*unit_coil_P0.diagonal()
jz_hdiv_P0 = (phiz_Hdiv_P0.T@j_hdiv)*unit_coil_P0.diagonal()

##############################################################################

##############################################################################

jx_L2 = -sigma*(dphix_H1.T@potential_H1)*unit_coil.diagonal()
jy_L2 = -sigma*(dphiy_H1.T@potential_H1)*unit_coil.diagonal()
jz_L2 = -sigma*(dphiz_H1.T@potential_H1)*unit_coil.diagonal()

dphix_H1_P0, dphiy_H1_P0, dphiz_H1_P0 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = 0)
jx_L2_P0 = -sigma*(dphix_H1_P0.T@potential_H1)*unit_coil_P0.diagonal()
jy_L2_P0 = -sigma*(dphiy_H1_P0.T@potential_H1)*unit_coil_P0.diagonal()
jz_L2_P0 = -sigma*(dphiz_H1_P0.T@potential_H1)*unit_coil_P0.diagonal()

phi_j = x

##############################################################################

# grid = pde.tools.vtklib.createVTK(MESH)
# pde.tools.vtklib.add_H1_Scalar(grid, potential_H1, 'potential_H1')
# pde.tools.vtklib.add_L2_Vector(grid,jx_L2_P0,jy_L2_P0,jz_L2_P0,'j_l2')
# pde.tools.add_L2_Vector(grid,jx_hdiv_P0,jy_hdiv_P0,jz_hdiv_P0,'j_hdiv')
# pde.tools.add_L2_Scalar(grid,potential_L2,'potential_L2')
# pde.tools.vtklib.writeVTK(grid, 'das2.vtu')