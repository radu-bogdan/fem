import ngsolve as ng
import netgen.occ as occ
# import netgen.gui
from netgen.webgui import Draw as DrawGeo
import numpy as np

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

geoOCCmesh = geoOCC.GenerateMesh()
geoOCCmesh.Refine()
geoOCCmesh.Refine()
# geoOCCmesh.Refine()


import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
from sksparse.cholmod import cholesky as chol

MESH = pde.mesh3.netgen(geoOCCmesh)

phi_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = 4)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = 4)
D = pde.int.assemble3(MESH, order = 4)

M = phi_H1 @ D @ phi_H1.T

K = dphix_H1 @ D @ dphix_H1.T +\
    dphiy_H1 @ D @ dphiy_H1.T +\
    dphiz_H1 @ D @ dphiz_H1.T

R0, RSS = pde.h1.assembleR3(MESH, space = 'P1', faces = 'ambient_face')

Kn = RSS @ K @ RSS.T






phix_Hcurl, phiy_Hcurl, phiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'M', order = 4)
curlphix_Hcurl, curlphiy_Hcurl, curlphiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'K', order = 4)

M_Hcurl = phix_Hcurl @ D @ phix_Hcurl.T +\
          phiy_Hcurl @ D @ phiy_Hcurl.T +\
          phiz_Hcurl @ D @ phiz_Hcurl.T
          

K_Hcurl = curlphix_Hcurl @ D @ curlphix_Hcurl.T +\
          curlphiy_Hcurl @ D @ curlphiy_Hcurl.T +\
          curlphiz_Hcurl @ D @ curlphiz_Hcurl.T




from mst import *

# newListOfEdges = MESH.EdgesToVertices[:,:2]
# newListOfEdges = np.r_[np.delete(MESH.EdgesToVertices[:,:2],MESH.Boundary_Edges,axis=0),
#                        MESH.EdgesToVertices[MESH.Boundary_Edges,:2]]

newListOfEdges = np.r_[MESH.EdgesToVertices[MESH.Boundary_Edges,:2],
                       np.delete(MESH.EdgesToVertices[:,:2],MESH.Boundary_Edges,axis=0)]

# newListOfEdges = np.delete(MESH.EdgesToVertices[:,:2],MESH.Boundary_Edges,axis=0)

nPe = np.unique(MESH.e[:,:2]).size

g = Graph(MESH.np) 

for i in range(MESH.NoEdges):
    g.addEdge(newListOfEdges[i,0],newListOfEdges[i,1],i)
    

g.KruskalMST()
indices = np.array(g.MST)[:,2]



    