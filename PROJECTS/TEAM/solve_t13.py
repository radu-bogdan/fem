from t13geo import *

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
from sksparse.cholmod import cholesky as chol
from scipy import sparse as sp
import numpy as npy

order = 1

MESH = pde.mesh3.netgen(geoOCCmesh)

phi_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = order)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = order)
D = pde.int.assemble3(MESH, order = order)

M = phi_H1 @ D @ phi_H1.T

K = dphix_H1 @ D @ dphix_H1.T +\
    dphiy_H1 @ D @ dphiy_H1.T +\
    dphiz_H1 @ D @ dphiz_H1.T

R0, RSS = pde.h1.assembleR3(MESH, space = 'P1', faces = 'ambient_face')

Kn = RSS @ K @ RSS.T

coeff = lambda x,y,z : 1+0*x*y*z
J = pde.int.evaluate3(MESH, order = order, coeff = coeff, regions = 'mid_steel').diagonal()

r = J @ D @ phi_H1.T

# # solve:
u_H1 = RSS.T@(chol(Kn).solve_A(RSS@r))
# u = chol(M).solve_A(r)



phix_Hcurl, phiy_Hcurl, phiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'M', order = order)
curlphix_Hcurl, curlphiy_Hcurl, curlphiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'K', order = order)

M_Hcurl = phix_Hcurl @ D @ phix_Hcurl.T +\
          phiy_Hcurl @ D @ phiy_Hcurl.T +\
          phiz_Hcurl @ D @ phiz_Hcurl.T
          

K_Hcurl = curlphix_Hcurl @ D @ curlphix_Hcurl.T +\
          curlphiy_Hcurl @ D @ curlphiy_Hcurl.T +\
          curlphiz_Hcurl @ D @ curlphiz_Hcurl.T


coeffx = lambda x,y,z : z
coeffy = lambda x,y,z : x
coeffz = lambda x,y,z : y

Jx = pde.int.evaluate3(MESH, order = order, coeff = coeffx, regions = 'mid_steel').diagonal()
Jy = pde.int.evaluate3(MESH, order = order, coeff = coeffy, regions = 'mid_steel').diagonal()
Jz = pde.int.evaluate3(MESH, order = order, coeff = coeffz, regions = 'mid_steel').diagonal()

r = Jx @ D @ phix_Hcurl.T +\
    Jy @ D @ phiy_Hcurl.T +\
    Jz @ D @ phiz_Hcurl.T


phix_Hcurl_P0, phiy_Hcurl_P0, phiz_Hcurl_P0 = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'M', order = 1)
u = pde.pcg(M_Hcurl,r)

ux = phix_Hcurl_P0.T @ u
uy = phiy_Hcurl_P0.T @ u
uz = phiz_Hcurl_P0.T @ u

normu = ux**2+uy**2+uz**2

MESH.pdesurf(u_H1, faces = 'l_steel_face,r_steel_face,mid_steel_face,coil_face')





import numpy as np
B = np.array([0, 0.0025, 0.0050, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80])
H = np.array([0, 16, 30, 54, 93, 143, 191, 210, 222, 233, 247, 258, 272, 289, 313, 342, 377, 433, 509, 648, 933, 1228, 1934, 2913, 4993, 7189, 9423])


##########################################################################
# Tree/Cotree gauging
##########################################################################

from mst import *

newListOfEdges = MESH.EdgesToVertices[:,:2]

g = Graph(MESH.np)

for i in range(newListOfEdges.shape[0]):
    g.addEdge(newListOfEdges[i,0],newListOfEdges[i,1],i)
    
g.KruskalMST()
indices = np.array(g.MST)[:,2]

LIST_DOF = np.setdiff1d(np.r_[:MESH.NoEdges],indices)

D = sp.eye(MESH.NoEdges, format = 'csc')

R = D[:,LIST_DOF]

KR = R.T@K_Hcurl@R

##############################################################################

J1 = lambda x,y,z : np.c_[ 1+0*x, 0*y, 0*z]
J2 = lambda x,y,z : np.c_[ 0*x, 1+0*y, 0*z]
J3 = lambda x,y,z : np.c_[-1+0*x, 0*y, 0*z]
J4 = lambda x,y,z : np.c_[ 0*x,-1+0*y, 0*z]


J = lambda x,y,z : np.tile(((x<75)*(x>-75)*(y>-95)*(y<-75)),(3,1)).T*J1(x,y,z) +\
                   np.tile(((x<75)*(x>-75)*(y> 75)*(y< 95)),(3,1)).T*J3(x,y,z) +\
                   np.tile(((x<-75)*(x>-95)*(y<75)*(y>-75)),(3,1)).T*J4(x,y,z) +\
                   np.tile(((x> 75)*(x< 95)*(y<75)*(y>-75)),(3,1)).T*J2(x,y,z)
                   
##############################################################################



import vtk

points = vtk.vtkPoints()
grid = vtk.vtkUnstructuredGrid()

for i in range(MESH.np):
    points.InsertPoint(i, (MESH.p[i,0], MESH.p[i,1], MESH.p[i,2]))
    
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

for elem in elems:
    grid.InsertNextCell(elem.GetCellType(), elem.GetPointIds())



scalars = MESH.t[:,-1]
pdata = grid.GetCellData()
data = vtk.vtkDoubleArray()
data.SetNumberOfValues(MESH.nt)
for i,p in enumerate(scalars): data.SetValue(i,p)
pdata.SetScalars(data)

vec = vtk.vtkFloatArray()
vec.SetNumberOfComponents(3)

for i in range(MESH.nt):
    Je = J(MESH.mp_tet[i,0],MESH.mp_tet[i,1],MESH.mp_tet[i,2])[0]
    vec.InsertNextTuple([Je[0],Je[1],Je[2]])
pdata.SetVectors(vec)




writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName("whatever.vkt")
writer.SetInputData(grid)
writer.Write()
