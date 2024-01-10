from t13geo import *

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
from sksparse.cholmod import cholesky as chol
from scipy import sparse as sp
import numpy as npy
import time
import vtk
from scipy.sparse import bmat


MESH = pde.mesh3.netgen(geoOCCmesh)

##############################################################################
# B-H curves
##############################################################################


import numpy as np
B = np.array([0, 0.0025, 0.0050, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80])
H = np.array([0, 16, 30, 54, 93, 143, 191, 210, 222, 233, 247, 258, 272, 289, 313, 342, 377, 433, 509, 648, 933, 1228, 1934, 2913, 4993, 7189, 9423])

##############################################################################
# Current density (approx)
##############################################################################

J1 = lambda x,y,z : np.c_[ 1+0*x,   0*y, 0*z]
J2 = lambda x,y,z : np.c_[   0*x, 1+0*y, 0*z]
J3 = lambda x,y,z : np.c_[-1+0*x,   0*y, 0*z]
J4 = lambda x,y,z : np.c_[   0*x,-1+0*y, 0*z]
JR = lambda x,y,z,m,n : np.c_[-(y-n)*1/np.sqrt((x-m)**2+(y-n)**2),
                               (x-m)*1/np.sqrt((x-m)**2+(y-n)**2), 
                                0*z]

J = lambda x,y,z : np.tile(((x<= 50)*(x>= -50)*(y< -75)*(y>=-100)*(z>-50)*(z<50)),(3,1)).T*J1(x,y,z) +\
                   np.tile(((x<= 50)*(x>= -50)*(y>= 75)*(y<= 100)*(z>-50)*(z<50)),(3,1)).T*J3(x,y,z) +\
                   np.tile(((x<=-75)*(x>=-100)*(y<= 50)*(y>= -50)*(z>-50)*(z<50)),(3,1)).T*J4(x,y,z) +\
                   np.tile(((x>= 75)*(x<= 100)*(y<= 50)*(y>= -50)*(z>-50)*(z<50)),(3,1)).T*J2(x,y,z) +\
                   np.tile(((x>= 50)*(x<= 100)*(y>= 50)*(y<= 100)*(z>-50)*(z<50)),(3,1)).T*JR(x,y,z, 50, 50) +\
                   np.tile(((x<=-50)*(x>=-100)*(y<=-50)*(y>=-100)*(z>-50)*(z<50)),(3,1)).T*JR(x,y,z,-50,-50) +\
                   np.tile(((x<=-50)*(x>=-100)*(y>= 50)*(y<= 100)*(z>-50)*(z<50)),(3,1)).T*JR(x,y,z,-50, 50) +\
                   np.tile(((x>= 50)*(x<= 100)*(y<=-50)*(y>=-100)*(z>-50)*(z<50)),(3,1)).T*JR(x,y,z, 50,-50)

##############################################################################
# Tree/Cotree gauging
##############################################################################

from mst import *

newListOfEdges = MESH.EdgesToVertices[:,:2]

g = Graph(MESH.np)

for i in range(newListOfEdges.shape[0]):
    g.addEdge(newListOfEdges[i,0],newListOfEdges[i,1],i)
    
g.KruskalMST()
indices = np.array(g.MST)[:,2]

LIST_DOF = np.setdiff1d(np.r_[:MESH.NoEdges],indices)

DD = sp.eye(MESH.NoEdges, format = 'csc')

R = DD[:,LIST_DOF]


##############################################################################
# Assembly
##############################################################################

order = 2

phi_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = order)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = order)
D = pde.int.assemble3(MESH, order = order)
R0, RSS = pde.h1.assembleR3(MESH, space = 'P1', faces = 'ambient_face')

M = phi_H1 @ D @ phi_H1.T

K = dphix_H1 @ D @ dphix_H1.T +\
    dphiy_H1 @ D @ dphiy_H1.T +\
    dphiz_H1 @ D @ dphiz_H1.T

Kn = RSS @ K @ RSS.T

phix_Hcurl, phiy_Hcurl, phiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'M', order = order)
curlphix_Hcurl, curlphiy_Hcurl, curlphiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'K', order = order)

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

eJx = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : J(x,y,z)[:,0], regions = 'coil').diagonal()
eJy = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : J(x,y,z)[:,1], regions = 'coil').diagonal()
eJz = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : J(x,y,z)[:,2], regions = 'coil').diagonal()

eJx0 = pde.int.evaluate3(MESH, order = 0, coeff = lambda x,y,z : J(x,y,z)[:,0], regions = 'coil').diagonal()
eJy0 = pde.int.evaluate3(MESH, order = 0, coeff = lambda x,y,z : J(x,y,z)[:,1], regions = 'coil').diagonal()
eJz0 = pde.int.evaluate3(MESH, order = 0, coeff = lambda x,y,z : J(x,y,z)[:,2], regions = 'coil').diagonal()

r = eJx @ D @ phix_Hcurl.T +\
    eJy @ D @ phiy_Hcurl.T +\
    eJz @ D @ phiz_Hcurl.T
    
##############################################################################
# Only coil stuff...
##############################################################################

    
unit_coil = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : 1+0*x, regions = 'coil')

K_Hcurl_coil_full = curlphix_Hcurl @ D @ unit_coil @ curlphix_Hcurl.T +\
                    curlphiy_Hcurl @ D @ unit_coil @ curlphiy_Hcurl.T +\
                    curlphiz_Hcurl @ D @ unit_coil @ curlphiz_Hcurl.T
          
C_Hcurl_H1_coil_full = phix_Hcurl @ D @ unit_coil @ dphix_H1.T +\
                       phiy_Hcurl @ D @ unit_coil @ dphiy_H1.T +\
                       phiz_Hcurl @ D @ unit_coil @ dphiz_H1.T
               
non_zero_rows_K = np.where((np.diff(K_Hcurl_coil_full.indptr) != 0))[0]
non_zero_rows_C = np.where((np.diff(C_Hcurl_H1_coil_full.indptr) != 0))[0]

K_Hcurl_coil = K_Hcurl_coil_full[:,non_zero_rows_K]
K_Hcurl_coil = K_Hcurl_coil[non_zero_rows_K,:]

C_Hcurl_H1_coil = C_Hcurl_H1_coil_full[non_zero_rows_K,:]
C_Hcurl_H1_coil = C_Hcurl_H1_coil[:,non_zero_rows_C]

AA = bmat([[K_Hcurl_coil, C_Hcurl_H1_coil], 
           [C_Hcurl_H1_coil.T, None]]).tocsc()

          
# MESH.pdesurf(u_H1, faces = 'l_steel_face,r_steel_face,mid_steel_face,coil_face')

##############################################################################
# Solving stuff...
##############################################################################
    
cholKR = chol(KR)
x = cholKR.solve_A(R.T@r)
x = R@x

ux = curlphix_Hcurl_P0.T @ x
uy = curlphiy_Hcurl_P0.T @ x
uz = curlphiz_Hcurl_P0.T @ x
# stop



C_Hcurl_H1_eich = C_Hcurl_H1[:,:-1] # removing last entry

AA = bmat([[K_Hcurl, C_Hcurl_H1_eich], 
           [C_Hcurl_H1_eich.T, None]]).tocsc()

bb = np.r_[r,np.zeros(MESH.np-1)]


tm = time.monotonic()
xx = sp.linalg.spsolve(AA,bb)
print('Solving saddle point took ... ',time.monotonic()-tm)


xx = xx[-MESH.np+1:]
xx = np.r_[xx,0] # readding the last entry

dx_xx = dphix_H1_P0.T@xx
dy_xx = dphiy_H1_P0.T@xx
dz_xx = dphiz_H1_P0.T@xx

eJx0_new = eJx0 - dx_xx
eJy0_new = eJy0 - dy_xx
eJz0_new = eJz0 - dz_xx

##############################################################################
# Storing to vtk
##############################################################################


tm = time.monotonic()

points = vtk.vtkPoints()
grid = vtk.vtkUnstructuredGrid()

for i in range(MESH.np): points.InsertPoint(i, (MESH.p[i,0], MESH.p[i,1], MESH.p[i,2]))
    
def create_cell(i):
    tetra = vtk.vtkTetra()
    ids = tetra.GetPointIds()
    ids.SetId(0, MESH.t[i,0])
    ids.SetId(1, MESH.t[i,1])
    ids.SetId(2, MESH.t[i,2])
    ids.SetId(3, MESH.t[i,3])
    return tetra

elems = [create_cell(i) for i in range(MESH.nt)]
grid.Allocate(MESH.nt, 1)
grid.SetPoints(points)

for elem in elems: grid.InsertNextCell(elem.GetCellType(), elem.GetPointIds())



scalars = MESH.t[:,-1]
pdata = grid.GetCellData()
data = vtk.vtkDoubleArray()
data.SetNumberOfValues(MESH.nt)
for i,p in enumerate(scalars): data.SetValue(i,p)
pdata.SetScalars(data)



vecJ = vtk.vtkFloatArray()
vecJ.SetNumberOfComponents(3)
for i in range(MESH.nt):
    # Je = J(MESH.mp_tet[i,0],MESH.mp_tet[i,1],MESH.mp_tet[i,2])[0]
    vecJ.InsertNextTuple([eJx0[i],eJy0[i],eJz0[i]])
vecJ.SetName('omg')
pdata.AddArray(vecJ)


vecJ = vtk.vtkFloatArray()
vecJ.SetNumberOfComponents(3)
for i in range(MESH.nt):
    # Je = J(MESH.mp_tet[i,0],MESH.mp_tet[i,1],MESH.mp_tet[i,2])[0]
    vecJ.InsertNextTuple([eJx0_new[i],eJy0_new[i],eJz0_new[i]])
vecJ.SetName('omg2')
pdata.AddArray(vecJ)



vec = vtk.vtkFloatArray()
vec.SetNumberOfComponents(3)
for i in range(MESH.nt):
    vec.InsertNextTuple([ux[i],uy[i],uz[i]])
vec.SetName('lel')
# pdata.SetVectors([vec,vecJ])
pdata.AddArray(vecJ)
print('Time needed to prepare file ... ',time.monotonic()-tm)

writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName("whatever.vkt")
writer.SetInputData(grid)
writer.Write()
print('Time needed to write to file ... ',time.monotonic()-tm)
