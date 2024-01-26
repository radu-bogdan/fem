import sys
sys.path.insert(0,'../../') # adds parent directory
from scipy import sparse as sp

from solve_t13_strom import *

MESH = pde.mesh3.netgen(geoOCCmesh)

##############################################################################
# B-H curves
##############################################################################

B = np.array([0, 0.0025, 0.0050, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80])
H = np.array([0, 16, 30, 54, 93, 143, 191, 210, 222, 233, 247, 258, 272, 289, 313, 342, 377, 433, 509, 648, 933, 1228, 1934, 2913, 4993, 7189, 9423])

##############################################################################
# Tree/Cotree gauging
##############################################################################

R = pde.tools.tree_cotree_gauge(MESH)

##############################################################################
# Assembly
##############################################################################

order = 1

phi_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = order)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = order)
D = pde.int.assemble3(MESH, order = order)


M = phi_H1 @ D @ phi_H1.T

K = dphix_H1 @ D @ dphix_H1.T +\
    dphiy_H1 @ D @ dphiy_H1.T +\
    dphiz_H1 @ D @ dphiz_H1.T

# Kn = RSS @ K @ RSS.T

phix_Hcurl, phiy_Hcurl, phiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'M', order = order)
curlphix_Hcurl, curlphiy_Hcurl, curlphiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'K', order = order)

R0, RSS = pde.hcurl.assembleR3(MESH, space = 'N0', faces = 'ambient_face')

M_Hcurl = phix_Hcurl @ D @ phix_Hcurl.T +\
          phiy_Hcurl @ D @ phiy_Hcurl.T +\
          phiz_Hcurl @ D @ phiz_Hcurl.T

K_Hcurl = curlphix_Hcurl @ D @ curlphix_Hcurl.T +\
          curlphiy_Hcurl @ D @ curlphiy_Hcurl.T +\
          curlphiz_Hcurl @ D @ curlphiz_Hcurl.T
          
C_Hcurl_H1 = phix_Hcurl @ D @ dphix_H1.T +\
             phiy_Hcurl @ D @ dphiy_H1.T +\
             phiz_Hcurl @ D @ dphiz_H1.T

curlphix_Hcurl_P0, curlphiy_Hcurl_P0, curlphiz_Hcurl_P0 = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'K', order = 0)
phix_Hcurl_P0, phiy_Hcurl_P0, phiz_Hcurl_P0 = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'M', order = 0)
dphix_H1_P0, dphiy_H1_P0, dphiz_H1_P0 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = 0)

KR = R.T@K_Hcurl@R
MR = R.T@M_Hcurl@R

r = dx_x @ D @ phix_Hcurl.T +\
    dy_x @ D @ phiy_Hcurl.T +\
    dz_x @ D @ phiz_Hcurl.T



##############################################################################
# Only coil stuff...
##############################################################################

    
# unit_coil = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : 1+0*x, regions = 'coil')
#
# K_Hcurl_coil_full = curlphix_Hcurl @ D @ unit_coil @ curlphix_Hcurl.T +\
#                     curlphiy_Hcurl @ D @ unit_coil @ curlphiy_Hcurl.T +\
#                     curlphiz_Hcurl @ D @ unit_coil @ curlphiz_Hcurl.T
#
# C_Hcurl_H1_coil_full = phix_Hcurl @ D @ unit_coil @ dphix_H1.T +\
#                        phiy_Hcurl @ D @ unit_coil @ dphiy_H1.T +\
#                        phiz_Hcurl @ D @ unit_coil @ dphiz_H1.T
#
# non_zero_rows_K = np.where((np.diff(K_Hcurl_coil_full.indptr) != 0))[0]
# non_zero_rows_C = np.where((np.diff(C_Hcurl_H1_coil_full.indptr) != 0))[0]
#
# K_Hcurl_coil = K_Hcurl_coil_full[:,non_zero_rows_K]
# K_Hcurl_coil = K_Hcurl_coil[non_zero_rows_K,:]
#
# C_Hcurl_H1_coil = C_Hcurl_H1_coil_full[non_zero_rows_K,:]
# C_Hcurl_H1_coil = C_Hcurl_H1_coil[:,non_zero_rows_C]
# C_Hcurl_H1_coil_eich = C_Hcurl_H1_coil[:,:-1]
#
# # Eliminate harmonic? seems to work but what do I kno
# K_Hcurl_coil_eich = K_Hcurl_coil[:,:-1]
# K_Hcurl_coil_eich = K_Hcurl_coil_eich[:-1,:]
# C_Hcurl_H1_coil_eich = C_Hcurl_H1_coil_eich[:-1,:]
#
# AA = bmat([[K_Hcurl_coil_eich, C_Hcurl_H1_coil_eich],
#            [C_Hcurl_H1_coil_eich.T, None]]).tocsc()
#
# r_coil = r[non_zero_rows_K]
# r_coil_eich = r_coil[:-1]
#
# bb = np.r_[r_coil_eich,np.zeros(non_zero_rows_C.size-1)]
#
# tm = time.monotonic()
# xxx = sp.linalg.spsolve(AA,bb)
# print('Solving saddle point took ... ',time.monotonic()-tm)
#
# xxx = xxx[-non_zero_rows_C.size+1:]
# xxx = np.r_[xxx,0] # readding the last entry
#
# xx = np.zeros(MESH.np)
# xx[non_zero_rows_C] = xxx

##############################################################################
# Solving stuff...
##############################################################################


# C_Hcurl_H1_eich = C_Hcurl_H1[:,:-1] # removing last entry

# AA = bmat([[K_Hcurl, C_Hcurl_H1_eich], 
#            [C_Hcurl_H1_eich.T, None]]).tocsc()

# bb = np.r_[r,np.zeros(MESH.np-1)]


# tm = time.monotonic()
# xx = sp.linalg.spsolve(AA,bb)
# print('Solving saddle point took ... ',time.monotonic()-tm)


# xx = xx[-MESH.np+1:]
# xx = np.r_[xx,0] # readding the last entry

##############################################################################

# dx_xx = dphix_H1.T@xx
# dy_xx = dphiy_H1.T@xx
# dz_xx = dphiz_H1.T@xx

# eJx_new = eJx - 0*dx_xx
# eJy_new = eJy - 0*dy_xx
# eJz_new = eJz - 0*dz_xx

##############################################################################

# r = eJx @ D @ phix_Hcurl.T +\
#     eJy @ D @ phiy_Hcurl.T +\
#     eJz @ D @ phiz_Hcurl.T

cholKR = chol(KR)
x = cholKR.solve_A(R.T@r)
# x = pde.pcg(KR,R.T@r,output=True,pfuns = lambda x : sp.spdiags(MR.diagonal(), 0,MR.shape)@x,maxit=10000)
# x = pde.pcg(KR,R.T@r,output=True,pfuns = lambda x : sp.spdiags(10**(-6)*(np.arange(MR.shape[0])+1), 0,MR.shape)@x,maxit=10000)
# x = pde.pcg(KR,R.T@r,output=True,maxit=1e14,tol=1e-14)
x = R@x



ux = curlphix_Hcurl_P0.T @ x
uy = curlphiy_Hcurl_P0.T @ x
uz = curlphiz_Hcurl_P0.T @ x

##############################################################################
# Storing to vtk
##############################################################################

import vtklib

grid = vtklib.createVTK(MESH)
vtklib.add_H1_Scalar(grid, phi_j, 'lel')
# vtklib.add_L2_Vector(grid,evJx,evJy,evJz,'kek')
vtklib.add_L2_Vector(grid,dx_x_P0,dy_x_P0,dz_x_P0,'kek2')
vtklib.add_L2_Vector(grid,ux,uy,uz,'kek3')

vtklib.add_L2_Scalar(grid,dx_x_P0**2+dy_x_P0**2+dz_x_P0**2,'kek2magn')
vtklib.add_L2_Scalar(grid,ux**2+uy**2+uz**2,'kek3norm')
vtklib.writeVTK(grid, 'das2.vtu')

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
#
# scalars = MESH.t[:,-1]
# pdata = grid.GetCellData()
# data = vtk.vtkDoubleArray()
# data.SetNumberOfValues(MESH.nt)
# for i,p in enumerate(scalars): data.SetValue(i,p)
# pdata.SetScalars(data)
#
#
#
# vecJ = vtk.vtkFloatArray()
# vecJ.SetNumberOfComponents(3)
# for i in range(MESH.nt):
#     # Je = J(MESH.mp_tet[i,0],MESH.mp_tet[i,1],MESH.mp_tet[i,2])[0]
#     vecJ.InsertNextTuple([eJx0[i],eJy0[i],eJz0[i]])
# vecJ.SetName('omg')
# pdata.AddArray(vecJ)
#
#
#
# dx_xx = dphix_H1_P0.T@xx; dy_xx = dphiy_H1_P0.T@xx; dz_xx = dphiz_H1_P0.T@xx
# eJx0_new = eJx0 - dx_xx; eJy0_new = eJy0 - dy_xx; eJz0_new = eJz0 - dz_xx
# vecJ = vtk.vtkFloatArray()
# vecJ.SetNumberOfComponents(3)
# for i in range(MESH.nt):
#     # Je = J(MESH.mp_tet[i,0],MESH.mp_tet[i,1],MESH.mp_tet[i,2])[0]
#     vecJ.InsertNextTuple([eJx0_new[i],eJy0_new[i],eJz0_new[i]])
# vecJ.SetName('omg2')
# pdata.AddArray(vecJ)
#
#
#
# vec = vtk.vtkFloatArray()
# vec.SetNumberOfComponents(3)
# for i in range(MESH.nt):
#     vec.InsertNextTuple([ux[i],uy[i],uz[i]])
# vec.SetName('lel')
# # pdata.SetVectors([vec,vecJ])
# pdata.AddArray(vec)
# print('Time needed to prepare file ... ',time.monotonic()-tm)
#
# writer = vtk.vtkXMLUnstructuredGridWriter()
# writer.SetFileName("whatever.vtk")
# writer.SetInputData(grid)
# writer.Write()
# print('Time needed to write to file ... ',time.monotonic()-tm)
