import ngsolve as ng
import netgen.occ as occ
# import netgen.gui
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

geoOCCmesh = geoOCC.GenerateMesh(maxh = 25)
# geoOCCmesh.Refine()

geoOCCmesh.SecondOrder()

npoints2D = geoOCCmesh.Elements2D().NumPy()['np'].max()
npoints3D = geoOCCmesh.Elements3D().NumPy()['np'].max()

p = geoOCCmesh.Coordinates()

t = npy.c_[geoOCCmesh.Elements3D().NumPy()['nodes'].astype(npy.uint64)[:,:npoints3D],
           geoOCCmesh.Elements3D().NumPy()['index'].astype(npy.uint64)]-1

f = npy.c_[geoOCCmesh.Elements2D().NumPy()['nodes'].astype(npy.uint64)[:,:npoints2D],
           geoOCCmesh.Elements2D().NumPy()['index'].astype(npy.uint64)]-1

e = npy.c_[geoOCCmesh.Elements1D().NumPy()['nodes'].astype(npy.uint64)[:,:((npoints2D+1)//2)],
           geoOCCmesh.Elements1D().NumPy()['index'].astype(npy.uint64)]-1

max_ed_index = geoOCCmesh.Elements1D().NumPy()['index'].astype(npy.uint64).max()
max_bc_index = geoOCCmesh.Elements2D().NumPy()['index'].astype(npy.uint64).max()
max_rg_index = geoOCCmesh.Elements3D().NumPy()['index'].astype(npy.uint64).max()

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde

order = 3

geoOCCmesh.SecondOrder()

MESH = pde.mesh3.netgen(geoOCCmesh)
print(MESH)




# def second_order_sort(t):
#     def f(x,y):
#         if x+y == 1: return 4
#         if x+y == 2: return 5
#         if x+y == 4: return 8
#         if x+y == 5: return 9
        
#         if x+y == 3 and x*y == 0: return 6
#         if x+y == 3 and x*y == 2: return 7
#         return 'error'
    
#     f = np.vectorize(f)
    
#     ts = npy.argsort(t[:,:4])
    
#     ts5  = f(ts[:,0],ts[:,1])
#     ts6  = f(ts[:,0],ts[:,2])
#     ts7  = f(ts[:,0],ts[:,3])
#     ts8  = f(ts[:,1],ts[:,2])
#     ts9  = f(ts[:,1],ts[:,3])
#     ts10 = f(ts[:,2],ts[:,3])
    
#     indices = np.c_[ts,ts5,ts6,ts7,ts8,ts9,ts10,10+0*ts10]
#     sorted_t = np.take_along_axis(t, indices, axis=1)
#     return sorted_t



# new_t = second_order_sort(t)
# # new_t = t

# p0 = p[new_t[:,0],:]
# p1 = p[new_t[:,1],:]
# p2 = p[new_t[:,2],:]
# p3 = p[new_t[:,3],:]

# p4 = p[new_t[:,4],:]
# p5 = p[new_t[:,5],:]
# p6 = p[new_t[:,6],:]
# p7 = p[new_t[:,7],:]
# p8 = p[new_t[:,8],:]
# p9 = p[new_t[:,9],:]

# p10= p[new_t[:,10],:]

# print((p4-1/2*(p0+p1)).max())
# print((p5-1/2*(p0+p2)).max())
# print((p6-1/2*(p0+p3)).max())
# print((p7-1/2*(p1+p2)).max())
# print((p8-1/2*(p1+p3)).max())
# print((p9-1/2*(p2+p3)).max())



# Test:






# stop

mesh = ng.Mesh(geoOCCmesh)
# mesh.Refine()


##########################################################################
# NGSOVLE STUFF...
##########################################################################

# fes = ng.H1(mesh, order = 0)
# fes = ng.HCurl(mesh, order = 0)
fes = ng.HDiv(mesh, order = 0)
print ("Hx dofs:", fes.ndof)
u,v = fes.TnT()

# bfa = ng.BilinearForm(ng.grad(u)*ng.grad(v)*ng.dx).Assemble()
bfa = ng.BilinearForm(u*v*ng.dx).Assemble()
# bfa = ng.BilinearForm(ng.curl(u)*ng.curl(v)*ng.dx).Assemble()
# bfa = ng.BilinearForm(ng.div(u)*ng.div(v)*ng.dx).Assemble()

rows,cols,vals = bfa.mat.COO()
A = sp.csr_matrix((vals,(rows,cols)))

##########################################################################


import sys
sys.path.insert(0,'../../') # adds parent directory
import pde

order = 3

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
phix_Hdiv, phiy_Hdiv, phiz_Hdiv = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'M', order = order)
divphi_Hdiv = pde.hdiv.assemble3(MESH, space = 'RT0', matrix = 'K', order = order)

M_Hcurl = phix_Hcurl @ D @ phix_Hcurl.T +\
          phiy_Hcurl @ D @ phiy_Hcurl.T +\
          phiz_Hcurl @ D @ phiz_Hcurl.T

K_Hcurl = curlphix_Hcurl @ D @ curlphix_Hcurl.T +\
          curlphiy_Hcurl @ D @ curlphiy_Hcurl.T +\
          curlphiz_Hcurl @ D @ curlphiz_Hcurl.T
          
M_Hdiv = phix_Hdiv @ D @ phix_Hdiv.T +\
         phiy_Hdiv @ D @ phiy_Hdiv.T +\
         phiz_Hdiv @ D @ phiz_Hdiv.T
          
K_Hdiv = divphi_Hdiv @ D @ divphi_Hdiv.T
          
C_Hcurl_H1 = phix_Hcurl @ D @ dphix_H1.T +\
             phiy_Hcurl @ D @ dphiy_H1.T +\
             phiz_Hcurl @ D @ dphiz_H1.T



from scipy.sparse import bmat

AA = bmat([[K_Hcurl, C_Hcurl_H1],
           [C_Hcurl_H1.T, None]])

##########################################################################
# Tree/Cotree gauging
##########################################################################

from pde.tools.mst import *

random = np.random.permutation(MESH.EdgesToVertices[:,:2].shape[0])
newListOfEdges = MESH.EdgesToVertices[random,:2]

print(random)

# newListOfEdges = MESH.EdgesToVertices[:,:2]

g = Graph(MESH.np)

for i in range(newListOfEdges.shape[0]):
    g.addEdge(newListOfEdges[i,0],newListOfEdges[i,1],i)
    
g.KruskalMST()
indices = np.array(g.MST)[:,2]



LIST_DOF = np.setdiff1d(np.r_[:MESH.NoEdges],random[indices])
# LIST_DOF = np.setdiff1d(np.r_[:MESH.NoEdges],indices)

D = sp.eye(MESH.NoEdges, format = 'csc')

R = D[:,LIST_DOF]

KR = R.T@K_Hcurl@R
print(np.linalg.matrix_rank(KR.A),KR.shape)

##########################################################################


MESH = pde.mesh3.netgen(geoOCCmesh)


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


FaceNormals = np.zeros((MESH.NoFaces,2))
# for i in range(MESH.NoFaces):
#     p1 = 
    # n = 
    
f = MESH.f; p = MESH.p;
f0 = f[:,0]; f1 = f[:,1]; f2 = f[:,2]
B00 = p[f0,0]; B01 = p[f1,0]; B02 = p[f2,0];
B10 = p[f0,1]; B11 = p[f1,1]; B12 = p[f2,1];
B20 = p[f0,2]; B21 = p[f1,2]; B22 = p[f2,2];

