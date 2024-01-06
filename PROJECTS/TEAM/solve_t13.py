from t13geo import *

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
from sksparse.cholmod import cholesky as chol

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

# MESH.pdesurf(u_H1, faces = 'l_steel_face,r_steel_face,mid_steel_face,coil_face')





import numpy as np
B = np.array([0, 0.0025, 0.0050, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80])
H = np.array([0, 16, 30, 54, 93, 143, 191, 210, 222, 233, 247, 258, 272, 289, 313, 342, 377, 433, 509, 648, 933, 1228, 1934, 2913, 4993, 7189, 9423])


# MST
import pandas as pd 
nPoints = MESH.np
newListOfEdges = MESH.EdgesToVertices[:,:2]

listOfEdges = []
for i in range(MESH.NoEdges):
    tempNewListOfEdges = np.delete(newListOfEdges, i - len(listOfEdges), axis=0)
    newPoints = pd.unique(tempNewListOfEdges.ravel()).size
    if newPoints == nPoints:
        listOfEdges.append(i)
        newListOfEdges = tempNewListOfEdges





from mst import *

newListOfEdges = MESH.EdgesToVertices[:,:2]
g = Graph(MESH.np) 

for i in range(MESH.NoEdges):
    g.addEdge(newListOfEdges[i,0],newListOfEdges[i,1],i)
    
g.KruskalMST()
indices = np.array(g.MST)[:,2]






LIST_DOF  = np.unique(MESH.FEMLISTS['N0']['B']['LIST_DOF'][indices])
LIST_DOF2 = np.setdiff1d(npy.arange(sizeM),LIST_DOF)

D = sp.eye(sizeM, format = 'csc')

if listDOF.size > 0:
    LIST_DOF = listDOF

R1 = D[:,LIST_DOF]
R2 = D[:,LIST_DOF2]

return R1.T.tocsc(),R2.T.tocsc()
