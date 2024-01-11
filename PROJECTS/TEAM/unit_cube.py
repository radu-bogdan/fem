import ngsolve as ng
import netgen.occ as occ
# import netgen.gui
from netgen.webgui import Draw as DrawGeo
import numpy as np
import numpy as npy

from scipy import sparse as sp

full = occ.Box(occ.Pnt(-100,-100,-50), occ.Pnt(100,100,50))

##########################################################################
# Identifications
##########################################################################

for face in full.faces: face.name = 'ambient_face'
for edge in full.edges: edge.name = 'ambient_edges'

full.mat("full")

##########################################################################
# Generating mesh...
##########################################################################

geoOCC = occ.OCCGeometry(full)
ng.Draw(geoOCC)

geoOCCmesh = geoOCC.GenerateMesh(maxh = 50)
# geoOCCmesh.Refine()

mesh = ng.Mesh(geoOCCmesh)
# mesh.Refine()


##########################################################################
# NGSOVLE STUFF...
##########################################################################

# fes = ng.H1(mesh, order = 0)
fes = ng.HCurl(mesh, order = 0)
# fes = ng.HDiv(mesh, order = 0)
print ("Hx dofs:", fes.ndof)
u,v = fes.TnT()

# bfa = ng.BilinearForm(ng.grad(u)*ng.grad(v)*ng.dx).Assemble()
# bfa = ng.BilinearForm(u*v*ng.dx).Assemble()
bfa = ng.BilinearForm(ng.curl(u)*ng.curl(v)*ng.dx).Assemble()
# bfa = ng.BilinearForm(ng.div(u)*ng.div(v)*ng.dx).Assemble()

rows,cols,vals = bfa.mat.COO()
A = sp.csr_matrix((vals,(rows,cols)))

##########################################################################


import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
from sksparse.cholmod import cholesky as chol

order = 2

MESH = pde.mesh3.netgen(geoOCCmesh)
print(MESH)

phi_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = order)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = order)
D = pde.int.assemble3(MESH, order = order)

M = phi_H1 @ D @ phi_H1.T

K = dphix_H1 @ D @ dphix_H1.T +\
    dphiy_H1 @ D @ dphiy_H1.T +\
    dphiz_H1 @ D @ dphiz_H1.T

R0, RSS = pde.h1.assembleR3(MESH, space = 'P1', faces = 'ambient_face')

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

from scipy.sparse import bmat

AA = bmat([[K_Hcurl, C_Hcurl_H1],
           [C_Hcurl_H1.T, None]])

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
##########################################################################










curlphix_Hcurl, curlphiy_Hcurl, curlphiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'K', order = order)



regions_indices = pde.tools.getIndices(MESH.regions_2d,'ambient_face')
face_indices = np.where(np.in1d(MESH.f[:,-1],regions_indices))
faces = MESH.f[face_indices]

edges_trigs = npy.r_[npy.c_[faces[:,0],faces[:,1]],
                     npy.c_[faces[:,0],faces[:,2]],
                     npy.c_[faces[:,1],faces[:,2]]]

edges_trigs_unique, je = npy.unique(edges_trigs, axis = 0, return_inverse = True)

edge_indices = pde.intersect2d(MESH.EdgesToVertices[:,:2],edges_trigs_unique)
other_edge_indices = np.setdiff1d(np.r_[:MESH.NoEdges],edge_indices)

noPoints_interior = np.unique(MESH.EdgesToVertices[other_edge_indices,:2]).size

print(edge_indices.shape)

# LIST_DOF = np.setdiff1d(np.r_[:MESH.NoEdges],edge_indices)
# D = sp.eye(MESH.NoEdges, format = 'csc')
# R = D[:,LIST_DOF]

# K_noEdges = R.T@K_Hcurl@R
# print(K_noEdges.shape,np.linalg.matrix_rank(K_noEdges.A,tol=1e-12))





# LIST_DOF  = np.unique(MESH.FEMLISTS['N0']['B']['LIST_DOF'][indices])
# LIST_DOF2 = np.setdiff1d(np.r_[:MESH.NoEdges],indices)




# print(KR.shape,np.linalg.matrix_rank(KR.A, tol=1e-10))
# print(K_Hcurl.shape,np.linalg.matrix_rank(K_Hcurl.A, tol=1e-10))
# return R1.T.tocsc(),R2.T.tocsc()
