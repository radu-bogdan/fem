print('solve_t13_mag_pot_lin')

import sys
sys.path.insert(0,'../../') # adds parent directory
from scipy import sparse as sp

from solve_t13_strom import *

MESH = pde.mesh3.netgen(geoOCCmesh)

# @profile
# def do():

##############################################################################
# Tree/Cotree gauging
##############################################################################

# edges = MESH.EdgesToVertices[ambient_edges_indices,:2]
# edges = MESH.EdgesToVertices
# R = pde.tools.tree_cotree_gauge(MESH, edges = edges)
R = pde.tools.tree_cotree_gauge(MESH, random_edges = True)

##############################################################################
# Assembly
##############################################################################

order = 0

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

# R0, RSS = pde.hcurl.assembleR3(MESH, space = 'N0', faces = 'ambient_face')

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

# r = jx_L2 @ D @ phix_Hcurl.T +\
#     jy_L2 @ D @ phiy_Hcurl.T +\
#     jz_L2 @ D @ phiz_Hcurl.T

r = jx_hdiv @ D @ phix_Hcurl.T +\
    jy_hdiv @ D @ phiy_Hcurl.T +\
    jz_hdiv @ D @ phiz_Hcurl.T

##############################################################################
# Only coil stuff...
##############################################################################

cholKR = chol(KR.tocsc())
A = cholKR.solve_A(R.T@r)
# x = pde.pcg(KR,R.T@r,output=True,pfuns = lambda x : sp.spdiags(MR.diagonal(), 0,MR.shape)@x,maxit=10000)
# x = pde.pcg(KR,R.T@r,output=True,pfuns = lambda x : sp.spdiags(10**(-6)*(np.arange(MR.shape[0])+1), 0,MR.shape)@x,maxit=10000)
# x = pde.pcg(KR,R.T@r,output=True,maxit=1e14,tol=1e-14)
A = R@A

Bx = curlphix_Hcurl_P0.T @ A
By = curlphiy_Hcurl_P0.T @ A
Bz = curlphiz_Hcurl_P0.T @ A

##############################################################################
# Storing to vtk
##############################################################################

grid = pde.tools.vtklib.createVTK(MESH)
pde.tools.add_H1_Scalar(grid, potential_H1, 'potential_H1')
pde.tools.add_L2_Vector(grid,jx_L2,jy_L2,jz_L2,'j_L2')
pde.tools.add_L2_Vector(grid,jx_hdiv_P0,jy_hdiv_P0,jz_hdiv_P0,'j_hdiv')
pde.tools.add_L2_Vector(grid,Bx,By,Bz,'B')
pde.tools.vtklib.writeVTK(grid, 'vector_potential.vtu')

# do()