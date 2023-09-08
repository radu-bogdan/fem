import sys
sys.path.insert(0,'../../') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import plotly.io as pio
pio.renderers.default = 'browser'
# import nonlinear_Algorithms
import numba as nb
from scipy.sparse import hstack,vstack


import matplotlib.pyplot as plt
import matplotlib
cmap = plt.cm.jet

##########################################################################################
# Loading mesh
##########################################################################################

motor_npz = np.load('../meshes/motor_pizza.npz', allow_pickle = True)

geoOCC = motor_npz['geoOCC'].tolist()
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

import ngsolve as ng
geoOCCmesh = geoOCC.GenerateMesh()
ngsolve_mesh = ng.Mesh(geoOCCmesh)
# ngsolve_mesh.Refine()
# ngsolve_mesh.Refine()

MESH = pde.mesh.netgen(ngsolve_mesh.ngmesh)

linear = '*air,*magnet,shaft_iron,*coil'
nonlinear = 'stator_iron,rotor_iron'
rotor = 'rotor_iron,*magnet,rotor_air,shaft_iron'
##########################################################################################

def makeIdentifications(MESH):

    a = np.array(MESH.geoOCCmesh.GetIdentifications())

    c0 = np.zeros(a.shape[0])
    c1 = np.zeros(a.shape[0])

    for i in range(a.shape[0]):
        point0 = MESH.p[a[i,0]-1]
        point1 = MESH.p[a[i,1]-1]

        c0[i] = point0[0]**2+point0[1]**2
        c1[i] = point1[0]**2+point1[1]**2

    ind0 = np.argsort(c0)

    aa = np.c_[a[ind0[:-1],0]-1,
               a[ind0[1: ],0]-1]

    edges0 = np.c_[a[ind0[:-1],0]-1,
                   a[ind0[1: ],0]-1]
    edges1 = np.c_[a[ind0[:-1],1]-1,
                   a[ind0[1: ],1]-1]

    edges0 = np.sort(edges0)
    edges1 = np.sort(edges1)

    edgecoord0 = np.zeros(edges0.shape[0],dtype=int)
    edgecoord1 = np.zeros(edges1.shape[0],dtype=int)

    for i in range(edges0.shape[0]):
        edgecoord0[i] = np.where(np.all(MESH.EdgesToVertices[:,:2]==edges0[i,:],axis=1))[0][0]
        edgecoord1[i] = np.where(np.all(MESH.EdgesToVertices[:,:2]==edges1[i,:],axis=1))[0][0]

    identification = np.c_[np.r_[a[ind0,0]-1,MESH.np + edgecoord0],
                           np.r_[a[ind0,1]-1,MESH.np + edgecoord1]]
    ident_points = np.c_[a[ind0,0]-1,
                         a[ind0,1]-1]
    ident_edges = np.c_[edgecoord0,
                        edgecoord1]
    return ident_points, ident_edges


ident_points, ident_edges = makeIdentifications(MESH)


##########################################################################################
# Assembling stuff
##########################################################################################

space_Vh = 'N0'
space_Qh = 'P0'
int_order = 4

tm = time.monotonic()

phi_Hcurl = lambda x : pde.hcurl.assemble(MESH, space = space_Vh, matrix = 'phi', order = x)
curlphi_Hcurl = lambda x : pde.hcurl.assemble(MESH, space = space_Vh, matrix = 'curlphi', order = x)
phi_L2 = lambda x : pde.l2.assemble(MESH, space = space_Qh, matrix = 'M', order = x)

D = lambda x : pde.int.assemble(MESH, order = x)

Mh = lambda x: phi_Hcurl(x)[0] @ D(x) @ phi_Hcurl(x)[0].T + \
               phi_Hcurl(x)[1] @ D(x) @ phi_Hcurl(x)[1].T

D1 = D(1); D2 = D(2); D4 = D(4); Mh1 = Mh(1); Mh2 = Mh(2)
D_int_order = D(int_order)

phi_L2_o1 = phi_L2(1)
curlphi_Hcurl_o1 = curlphi_Hcurl(1)

phix_Hcurl = phi_Hcurl(int_order)[0]
phiy_Hcurl = phi_Hcurl(int_order)[1]


C = phi_L2(int_order) @ D(int_order) @ curlphi_Hcurl(int_order).T
Z = sps.csc_matrix((C.shape[0],C.shape[0]))

Ja = 0; J0 = 0
for i in range(48):
    Ja += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : j3[i], regions ='coil'+str(i+1)).diagonal()
    J0 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : j3[i], regions = 'coil'+str(i+1)).diagonal()
Ja = 0*Ja; J0 = 0*J0

M0 = 0; M1 = 0; M00 = 0; M10 = 0
for i in range(16):
    M0 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
    M1 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
    
    M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
    M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()

aM = phix_Hcurl@ D(int_order) @(M0) +\
     phiy_Hcurl@ D(int_order) @(M1)

aJ = phi_L2(int_order)@ D(int_order) @Ja

##########################################################################################

R_AL, R_ALR = pde.hcurl.assembleR(MESH, space = 'N0', edges = 'airL', listDOF = ident_edges[:,0])
R_AR, R_ARR = pde.hcurl.assembleR(MESH, space = 'N0', edges = 'airR', listDOF = ident_edges[:,1])

##########################################################################################

# iMh = pde.tools.fastBlockInverse(Mh1)
# S = C@iMh@C.T
# r = C@(iMh@aM)


# tm = time.monotonic(); x = sps.linalg.spsolve(S,r); print('dual: ',time.monotonic()-tm)
# MESH.pdesurf2(x)


from scipy.sparse import bmat

SYS = bmat([[Mh2,C.T],\
            [C,None]]).tocsc()

rhs = np.r_[aM,np.zeros(MESH.nt)]

tm = time.monotonic(); x2 = sps.linalg.spsolve(SYS,rhs); print('mixed: ',time.monotonic()-tm)
y2 = x2[MESH.NoEdges:]
MESH.pdesurf2(y2)


