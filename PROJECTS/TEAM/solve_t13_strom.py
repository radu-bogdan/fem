from t13geo import *

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
from sksparse.cholmod import cholesky as chol
from scipy import sparse as sp
import numpy as np
import time
import vtk
from scipy.sparse import bmat,hstack,vstack


MESH = pde.mesh3.netgen(geoOCCmesh)

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

evJ = J(MESH.mp_tet[:,0],MESH.mp_tet[:,1],MESH.mp_tet[:,2])

evJx = evJ[:,0]
evJy = evJ[:,1]
evJz = evJ[:,2]

###########################################################################

order = 1
D = pde.int.assemble3(MESH, order = order)
DB = pde.int.assembleB3(MESH, order = order)
unit_coil = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : 1+0*x, regions = 'coil')
coil_cut  = pde.int.evaluateB3(MESH, order = order, coeff = lambda x,y,z : 1+0*x, faces = 'coil_cut_1').diagonal()

###########################################################################

tm = time.monotonic()

phi_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = order)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = order)
phiB_H1 = pde.h1.assembleB3(MESH, space = 'P1', matrix = 'M', shape = phi_H1.shape, order = order)

R0, RSS = pde.h1.assembleR3(MESH, space = 'P1', faces = 'coil_cut_1')
# RSS = RSS[3:,:]
RSS = vstack((RSS,R0[1:,:]))


r = coil_cut @ DB @ phiB_H1.T

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

print(x.max())

###########################################################################

import vtklib

grid = vtklib.createVTK(MESH)
vtklib.add_H1_Scalar(grid, x, 'lel')
vtklib.writeVTK(grid, 'das2.vtu')


##############################################################################

stop

##############################################################################
# Hdiv proj
##############################################################################

order = 2
phix_Hdiv, phiy_Hdiv, phiz_Hdiv = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'M', order = order)
divphi_Hdiv = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'K', order = order)

phix_Hcurl, phiy_Hcurl, phiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'M', order = order)

unit_coil = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : 1+0*x, regions = 'coil')
phi_L2 = pde.l2.assemble3(MESH, space = 'P0', matrix = 'M', order = order)

D = pde.int.assemble3(MESH, order = order)

M_Hdiv = phix_Hdiv @ D @ phix_Hdiv.T +\
         phiy_Hdiv @ D @ phiy_Hdiv.T +\
         phiz_Hdiv @ D @ phiz_Hdiv.T
         
K_Hdiv = divphi_Hdiv @ D @ divphi_Hdiv.T

C_Hdiv_L2 = divphi_Hdiv @ D @ phi_L2.T

M_Hcurl = phix_Hcurl @ D @ unit_coil @ phix_Hcurl.T +\
          phiy_Hcurl @ D @ unit_coil @ phiy_Hcurl.T +\
          phiz_Hcurl @ D @ unit_coil @ phiz_Hcurl.T

M_Hdiv_coil_full = phix_Hdiv @ D @ unit_coil @ phix_Hdiv.T +\
                   phiy_Hdiv @ D @ unit_coil @ phiy_Hdiv.T +\
                   phiz_Hdiv @ D @ unit_coil @ phiz_Hdiv.T

M_Hcurl_coil_full = phix_Hcurl @ D @ unit_coil @ phix_Hcurl.T +\
                    phiy_Hcurl @ D @ unit_coil @ phiy_Hcurl.T +\
                    phiz_Hcurl @ D @ unit_coil @ phiz_Hcurl.T

RZdiv = pde.tools.removeZeros(M_Hdiv_coil_full)
M_Hdiv_coil = RZdiv @ M_Hdiv_coil_full @ RZdiv.T

RZcurl = pde.tools.removeZeros(M_Hcurl_coil_full)
M_Hcurl_coil = RZcurl @ M_Hcurl_coil_full @ RZcurl.T

##############################################################################

eJx = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : J(x,y,z)[:,0], regions = 'coil').diagonal()
eJy = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : J(x,y,z)[:,1], regions = 'coil').diagonal()
eJz = pde.int.evaluate3(MESH, order = order, coeff = lambda x,y,z : J(x,y,z)[:,2], regions = 'coil').diagonal()

r_hdiv = eJx @ D @ phix_Hdiv.T +\
         eJy @ D @ phiy_Hdiv.T +\
         eJz @ D @ phiz_Hdiv.T

r_hcurl = eJx @ D @ phix_Hcurl.T +\
          eJy @ D @ phiy_Hcurl.T +\
          eJz @ D @ phiz_Hcurl.T

A = bmat([[M_Hdiv,-C_Hdiv_L2],
          [C_Hdiv_L2.T, None]])
b = np.r_[r_hdiv,np.zeros(MESH.nt)]

x = sp.linalg.spsolve(A,b)
newJ = x[:MESH.NoFaces]

newJ_coil = sp.linalg.spsolve(M_Hdiv_coil,RZdiv @ r_hdiv)
# newJ_coil = sp.linalg.spsolve(M_Hcurl_coil,r_hdiv[non_zero_rows_Hcurl])

phix_Hdiv_P0, phiy_Hdiv_P0, phiz_Hdiv_P0 = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'M', order = 0)
phix_Hcurl_P0, phiy_Hcurl_P0, phiz_Hcurl_P0 = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'M', order = 0)


# newJ = np.zeros(MESH.NoFaces)
# newJ[non_zero_rows_Hdiv] = newJ_coil

newJx = phix_Hdiv_P0.T @ newJ
newJy = phiy_Hdiv_P0.T @ newJ
newJz = phiz_Hdiv_P0.T @ newJ

# newJ = np.zeros(MESH.NoEdges)
# newJ[non_zero_rows_Hcurl] = newJ_coil

# newJx = phix_Hcurl_P0.T @ newJ
# newJy = phiy_Hcurl_P0.T @ newJ
# newJz = phiz_Hcurl_P0.T @ newJ


##########################################################################
# NGSOVLE STUFF...
##########################################################################

import ngsolve as ng

mesh = ng.Mesh(geoOCCmesh)

# fes = ng.H1(mesh, order = 0)
fes = ng.HCurl(mesh, order = 0)
# fes = ng.HDiv(mesh, order = 0)
print ("Hx dofs:", fes.ndof)
u,v = fes.TnT()

# bfa = ng.BilinearForm(ng.grad(u)*ng.grad(v)*ng.dx).Assemble()
bfa = ng.BilinearForm(u*v*ng.dx).Assemble()
# bfa = ng.BilinearForm(ng.curl(u)*ng.curl(v)*ng.dx).Assemble()
# bfa = ng.BilinearForm(ng.div(u)*ng.div(v)*ng.dx).Assemble()

rows,cols,vals = bfa.mat.COO()
A = sp.csr_matrix((vals,(rows,cols)))

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
    vecJ.InsertNextTuple([newJx[i],newJy[i],newJz[i]])
vecJ.SetName('omg')
pdata.AddArray(vecJ)


vecJ = vtk.vtkFloatArray()
vecJ.SetNumberOfComponents(3)
for i in range(MESH.nt):
    vecJ.InsertNextTuple([evJx[i],evJy[i],evJz[i]])
vecJ.SetName('omg2')
pdata.AddArray(vecJ)


print('Time needed to prepare file ... ',time.monotonic()-tm)

writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName("whatever.vtu")
writer.SetInputData(grid)
writer.Write()
print('Time needed to write to file ... ',time.monotonic()-tm)


import vtklib

grid = vtklib.createVTK(MESH)
vtklib.add_H1_Scalar(grid, x, 'lel')
vtklib.writeVTK(grid, 'das.vtu')