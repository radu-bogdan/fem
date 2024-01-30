from t13geo import *

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
from sksparse.cholmod import cholesky as chol
import numpy as np
import time
import scipy.sparse as sp

MESH = pde.mesh3.netgen(geoOCCmesh)
print(MESH)
#########
# Checks:
ind_faces1 = pde.tools.getIndices(MESH.regions_2d,'coil_face')
ind_faces2 = pde.tools.getIndices(MESH.regions_2d,'coil_cut_1')
ind_faces3 = pde.tools.getIndices(MESH.regions_2d,'coil_cut_1,coil_face')

indices1 = np.in1d(MESH.BoundaryFaces_Region,ind_faces1)
indices2 = np.in1d(MESH.BoundaryFaces_Region,ind_faces2)
indices3 = np.in1d(MESH.BoundaryFaces_Region,ind_faces3)

dphix_H1, dphiy_H1, dphiz_H1 = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'M', order = 1)
sizeM = MESH.FEMLISTS['RT0']['TET']['sizeM']
LIST_DOF10  = np.unique(MESH.FEMLISTS['RT0']['B']['LIST_DOF'][indices1,:])
LIST_DOF20  = np.unique(MESH.FEMLISTS['RT0']['B']['LIST_DOF'][indices2,:])
LIST_DOF30  = np.unique(MESH.FEMLISTS['RT0']['B']['LIST_DOF'][indices3,:])


print(LIST_DOF10.shape,LIST_DOF20.shape,LIST_DOF30.shape)
# stop
#########


##############################################################################

face_index = pde.tools.getIndices(MESH.regions_2d,'coil_cut_1')
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
# stop
MESH = pde.mesh3(p_new,MESH.e,f_new,t_new,MESH.regions_3d,regions_2d_new,MESH.regions_1d,identifications = identifications)
print(MESH)

########
# Checks:
ind_faces1 = pde.tools.getIndices(MESH.regions_2d,'coil_face')
ind_faces2 = pde.tools.getIndices(MESH.regions_2d,'coil_cut_1')
ind_faces3 = pde.tools.getIndices(MESH.regions_2d,'coil_cut_1,new')

indices1 = np.in1d(MESH.BoundaryFaces_Region,ind_faces1)
indices2 = np.in1d(MESH.BoundaryFaces_Region,ind_faces2)
indices3 = np.in1d(MESH.BoundaryFaces_Region,ind_faces3)

dphix_H1, dphiy_H1, dphiz_H1 = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'M', order = 1)
sizeM = MESH.FEMLISTS['RT0']['TET']['sizeM']
LIST_DOF11  = np.unique(MESH.FEMLISTS['RT0']['B']['LIST_DOF'][indices1,:])
LIST_DOF21  = np.unique(MESH.FEMLISTS['RT0']['B']['LIST_DOF'][indices2,:])
LIST_DOF31  = np.unique(MESH.FEMLISTS['RT0']['B']['LIST_DOF'][indices3,:])


print(LIST_DOF11.shape,LIST_DOF21.shape,LIST_DOF31.shape)
# stop
########

##############################################################################
# Current density (approx)
##############################################################################

# J1 = lambda x,y,z : np.c_[ 1+0*x,   0*y, 0*z]
# J2 = lambda x,y,z : np.c_[   0*x, 1+0*y, 0*z]
# J3 = lambda x,y,z : np.c_[-1+0*x,   0*y, 0*z]
# J4 = lambda x,y,z : np.c_[   0*x,-1+0*y, 0*z]
# JR = lambda x,y,z,m,n : np.c_[-(y-n)*1/np.sqrt((x-m)**2+(y-n)**2),
#                                (x-m)*1/np.sqrt((x-m)**2+(y-n)**2),
#                                 0*z]
#
# J = lambda x,y,z : np.tile(((x<= 50)*(x>= -50)*(y< -75)*(y>=-100)*(z>-50)*(z<50)),(3,1)).T*J1(x,y,z) +\
#                    np.tile(((x<= 50)*(x>= -50)*(y>= 75)*(y<= 100)*(z>-50)*(z<50)),(3,1)).T*J3(x,y,z) +\
#                    np.tile(((x<=-75)*(x>=-100)*(y<= 50)*(y>= -50)*(z>-50)*(z<50)),(3,1)).T*J4(x,y,z) +\
#                    np.tile(((x>= 75)*(x<= 100)*(y<= 50)*(y>= -50)*(z>-50)*(z<50)),(3,1)).T*J2(x,y,z) +\
#                    np.tile(((x>= 50)*(x<= 100)*(y>= 50)*(y<= 100)*(z>-50)*(z<50)),(3,1)).T*JR(x,y,z, 50, 50) +\
#                    np.tile(((x<=-50)*(x>=-100)*(y<=-50)*(y>=-100)*(z>-50)*(z<50)),(3,1)).T*JR(x,y,z,-50,-50) +\
#                    np.tile(((x<=-50)*(x>=-100)*(y>= 50)*(y<= 100)*(z>-50)*(z<50)),(3,1)).T*JR(x,y,z,-50, 50) +\
#                    np.tile(((x>= 50)*(x<= 100)*(y<=-50)*(y>=-100)*(z>-50)*(z<50)),(3,1)).T*JR(x,y,z, 50,-50)
#
# evJ = J(MESH.mp_tet[:,0],MESH.mp_tet[:,1],MESH.mp_tet[:,2])
#
# evJx = evJ[:,0]; evJy = evJ[:,1]; evJz = evJ[:,2]

##############################################################################

order = 1
D = pde.int.assemble3(MESH, order = order)
DB = pde.int.assembleB3(MESH, order = order)
N1,N2,N3 = pde.int.assembleN3(MESH, order = order)
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

r = -RS0 @ K @ R1.T @ (1+np.zeros(R1.shape[0]))
K = RS0 @ K @ RS0.T

RZ = pde.tools.removeZeros(K)
K = RZ @ K @ RZ.T

r = RZ @ r

# stop

sigma = 1#58.7e6
x = chol(sigma*K).solve_A(r)
x = RS0.T @ RZ.T @ x + R1.T @ (1+np.zeros(R1.shape[0]))
print('My code took ... ',time.monotonic()-tm)

# stop

##############################################################################
# Hdiv solver
##############################################################################

order = 2
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

# print(RS1.shape,RZdiv.shape)

phiB_Hdiv = pde.hdiv.assembleB3(MESH, space = 'RT0', matrix = 'phi', shape = phix_Hdiv.shape, order = order)
unit_coil_B = pde.int.evaluateB3(MESH, order = order, coeff = lambda x,y,z : 1+0*x, faces = 'new').diagonal()
DB = pde.int.assembleB3(MESH, order = order)


# rhs= (face_in_1*N1 + face_in_2*N2 + face_in_3*N3) @ DB @ phiB_Hdiv.T

rhs = unit_coil_B @ DB @ phiB_Hdiv.T
# print(rhs[rhs!=0],np.argwhere(rhs!=0))
# rhs[np.argwhere(rhs!=0)[14]]=-0.5

rhs = np.r_[RS1@rhs,np.zeros(MESH.nt)]
# print(rhs[rhs!=0],np.argwhere(rhs!=0))

rhs = RZdiv @ rhs


xx = sp.linalg.spsolve(AA,rhs)

potential = (RZdiv.T@xx)[-MESH.nt:]
j_hdiv = RS1.T@(RZdiv.T@xx)[:-MESH.nt]


##############################################################################
unit_coil_P0 = pde.int.evaluate3(MESH, order = 0, coeff = lambda x,y,z : 1+0*x, regions = 'coil')

phix_Hdiv_P0, phiy_Hdiv_P0, phiz_Hdiv_P0 = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'M', order = 0)
jx_hdiv = (phix_Hdiv_P0.T@j_hdiv)*unit_coil_P0.diagonal()
jy_hdiv = (phiy_Hdiv_P0.T@j_hdiv)*unit_coil_P0.diagonal()
jz_hdiv = (phiz_Hdiv_P0.T@j_hdiv)*unit_coil_P0.diagonal()

##############################################################################

##############################################################################

dx_x = -(dphix_H1.T@x)*unit_coil.diagonal()
dy_x = -(dphiy_H1.T@x)*unit_coil.diagonal()
dz_x = -(dphiz_H1.T@x)*unit_coil.diagonal()

dphix_H1_P0, dphiy_H1_P0, dphiz_H1_P0 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = 0)
dx_x_P0 = -(dphix_H1_P0.T@x)*unit_coil_P0.diagonal()
dy_x_P0 = -(dphiy_H1_P0.T@x)*unit_coil_P0.diagonal()
dz_x_P0 = -(dphiz_H1_P0.T@x)*unit_coil_P0.diagonal()

phi_j = x

##############################################################################


import vtklib

grid = vtklib.createVTK(MESH)
vtklib.add_H1_Scalar(grid, x, 'lel')
# vtklib.add_L2_Vector(grid,evJx,evJy,evJz,'kek')
vtklib.add_L2_Vector(grid,dx_x_P0,dy_x_P0,dz_x_P0,'j_l2')
vtklib.add_L2_Vector(grid,jx_hdiv,jy_hdiv,jz_hdiv,'j_hdiv')

vtklib.add_L2_Scalar(grid,dx_x_P0**2+dy_x_P0**2+dz_x_P0**2,'kek2magn')
vtklib.add_L2_Scalar(grid,potential,'potential')
vtklib.writeVTK(grid, 'das2.vtu')

# import vtk.web
# import pyvista as pv
# mesh = pv.read('current_density.vtu')
# mesh.plot()

# ##############################################################################
#
# stop
#
# ##############################################################################
# # Hdiv proj
# ##############################################################################
#
# order = 2
# phix_Hdiv, phiy_Hdiv, phiz_Hdiv = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'M', order = order)
# divphi_Hdiv = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'K', order = order)
#
# phix_Hcurl, phiy_Hcurl, phiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'M', order = order)
#
# unit_coil = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : 1+0*x, regions = 'coil')
# phi_L2 = pde.l2.assemble3(MESH, space = 'P0', matrix = 'M', order = order)
#
# D = pde.int.assemble3(MESH, order = order)
#
# M_Hdiv = phix_Hdiv @ D @ phix_Hdiv.T +\
#          phiy_Hdiv @ D @ phiy_Hdiv.T +\
#          phiz_Hdiv @ D @ phiz_Hdiv.T
#
# K_Hdiv = divphi_Hdiv @ D @ divphi_Hdiv.T
#
# C_Hdiv_L2 = divphi_Hdiv @ D @ phi_L2.T
#
# M_Hcurl = phix_Hcurl @ D @ unit_coil @ phix_Hcurl.T +\
#           phiy_Hcurl @ D @ unit_coil @ phiy_Hcurl.T +\
#           phiz_Hcurl @ D @ unit_coil @ phiz_Hcurl.T
#
# M_Hdiv_coil_full = phix_Hdiv @ D @ unit_coil @ phix_Hdiv.T +\
#                    phiy_Hdiv @ D @ unit_coil @ phiy_Hdiv.T +\
#                    phiz_Hdiv @ D @ unit_coil @ phiz_Hdiv.T
#
# M_Hcurl_coil_full = phix_Hcurl @ D @ unit_coil @ phix_Hcurl.T +\
#                     phiy_Hcurl @ D @ unit_coil @ phiy_Hcurl.T +\
#                     phiz_Hcurl @ D @ unit_coil @ phiz_Hcurl.T
#
# RZdiv = pde.tools.removeZeros(M_Hdiv_coil_full)
# M_Hdiv_coil = RZdiv @ M_Hdiv_coil_full @ RZdiv.T
#
# RZcurl = pde.tools.removeZeros(M_Hcurl_coil_full)
# M_Hcurl_coil = RZcurl @ M_Hcurl_coil_full @ RZcurl.T
#
# ##############################################################################
#
# eJx = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : J(x,y,z)[:,0], regions = 'coil').diagonal()
# eJy = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : J(x,y,z)[:,1], regions = 'coil').diagonal()
# eJz = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : J(x,y,z)[:,2], regions = 'coil').diagonal()
#
# r_hdiv = eJx @ D @ phix_Hdiv.T +\
#          eJy @ D @ phiy_Hdiv.T +\
#          eJz @ D @ phiz_Hdiv.T
#
# r_hcurl = eJx @ D @ phix_Hcurl.T +\
#           eJy @ D @ phiy_Hcurl.T +\
#           eJz @ D @ phiz_Hcurl.T
#
# A = bmat([[M_Hdiv,-C_Hdiv_L2],
#           [C_Hdiv_L2.T, None]])
# b = np.r_[r_hdiv,np.zeros(MESH.nt)]
#
# x = sp.linalg.spsolve(A,b)
# newJ = x[:MESH.NoFaces]
#
# newJ_coil = sp.linalg.spsolve(M_Hdiv_coil,RZdiv @ r_hdiv)
# # newJ_coil = sp.linalg.spsolve(M_Hcurl_coil,r_hdiv[non_zero_rows_Hcurl])
#
# phix_Hdiv_P0, phiy_Hdiv_P0, phiz_Hdiv_P0 = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'M', order = 0)
# phix_Hcurl_P0, phiy_Hcurl_P0, phiz_Hcurl_P0 = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'M', order = 0)
#
#
# # newJ = np.zeros(MESH.NoFaces)
# # newJ[non_zero_rows_Hdiv] = newJ_coil
#
# newJx = phix_Hdiv_P0.T @ newJ
# newJy = phiy_Hdiv_P0.T @ newJ
# newJz = phiz_Hdiv_P0.T @ newJ
#
# # newJ = np.zeros(MESH.NoEdges)
# # newJ[non_zero_rows_Hcurl] = newJ_coil
#
# # newJx = phix_Hcurl_P0.T @ newJ
# # newJy = phiy_Hcurl_P0.T @ newJ
# # newJz = phiz_Hcurl_P0.T @ newJ
#
#
# ##########################################################################
# # NGSOVLE STUFF...
# ##########################################################################
#
# import ngsolve as ng
#
# mesh = ng.Mesh(geoOCCmesh)
#
# # fes = ng.H1(mesh, order = 0)
# fes = ng.HCurl(mesh, order = 0)
# # fes = ng.HDiv(mesh, order = 0)
# print ("Hx dofs:", fes.ndof)
# u,v = fes.TnT()
#
# # bfa = ng.BilinearForm(ng.grad(u)*ng.grad(v)*ng.dx).Assemble()
# bfa = ng.BilinearForm(u*v*ng.dx).Assemble()
# # bfa = ng.BilinearForm(ng.curl(u)*ng.curl(v)*ng.dx).Assemble()
# # bfa = ng.BilinearForm(ng.div(u)*ng.div(v)*ng.dx).Assemble()
#
# rows,cols,vals = bfa.mat.COO()
# A = sp.csr_matrix((vals,(rows,cols)))
#
# ##############################################################################
# # Storing to vtk
# ##############################################################################
#
#
# tm = time.monotonic()
#
# points = vtk.vtkPoints()
# grid = vtk.vtkUnstructuredGrid()
#
# for i in range(MESH.np): points.InsertPoint(i, (MESH.p[i,0], MESH.p[i,1], MESH.p[i,2]))
#
# def create_cell(i):
#     tetra = vtk.vtkTetra()
#     ids = tetra.GetPointIds()
#     ids.SetId(0, MESH.t[i,0])
#     ids.SetId(1, MESH.t[i,1])
#     ids.SetId(2, MESH.t[i,2])
#     ids.SetId(3, MESH.t[i,3])
#     return tetra
#
# elems = [create_cell(i) for i in range(MESH.nt)]
# grid.Allocate(MESH.nt, 1)
# grid.SetPoints(points)
#
# for elem in elems: grid.InsertNextCell(elem.GetCellType(), elem.GetPointIds())
#
#
# scalars = MESH.t[:,-1]
# pdata = grid.GetCellData()
# data = vtk.vtkDoubleArray()
# data.SetNumberOfValues(MESH.nt)
# for i,p in enumerate(scalars): data.SetValue(i,p)
# pdata.SetScalars(data)
#
#
# vecJ = vtk.vtkFloatArray()
# vecJ.SetNumberOfComponents(3)
# for i in range(MESH.nt):
#     vecJ.InsertNextTuple([newJx[i],newJy[i],newJz[i]])
# vecJ.SetName('omg')
# pdata.AddArray(vecJ)
#
#
# vecJ = vtk.vtkFloatArray()
# vecJ.SetNumberOfComponents(3)
# for i in range(MESH.nt):
#     vecJ.InsertNextTuple([evJx[i],evJy[i],evJz[i]])
# vecJ.SetName('omg2')
# pdata.AddArray(vecJ)
#
#
# print('Time needed to prepare file ... ',time.monotonic()-tm)
#
# writer = vtk.vtkXMLUnstructuredGridWriter()
# writer.SetFileName("whatever.vtu")
# writer.SetInputData(grid)
# writer.Write()
# print('Time needed to write to file ... ',time.monotonic()-tm)
#
#
# import vtklib
#
# grid = vtklib.createVTK(MESH)
# vtklib.add_H1_Scalar(grid, x, 'lel')
# vtklib.writeVTK(grid, 'das.vtu')