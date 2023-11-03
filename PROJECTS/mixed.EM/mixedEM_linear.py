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
from sksparse.cholmod import cholesky as chol


import matplotlib.pyplot as plt
import matplotlib
cmap = plt.cm.jet
import dill
import pickle

##########################################################################################
# Loading mesh
##########################################################################################
# motor_stator_npz = np.load('meshes/motor_stator.npz', allow_pickle = True)

# p_stator = motor_stator_npz['p'].T
# e_stator = motor_stator_npz['e'].T
# t_stator = motor_stator_npz['t'].T
# q_stator = np.empty(0)
# regions_2d_stator = motor_stator_npz['regions_2d']
# regions_1d_stator = motor_stator_npz['regions_1d']
# m = motor_stator_npz['m']; m_new = m
# j3 = motor_stator_npz['j3']

# motor_rotor_npz = np.load('meshes/motor_rotor.npz', allow_pickle = True)

# p_rotor = motor_rotor_npz['p'].T
# e_rotor = motor_rotor_npz['e'].T
# t_rotor = motor_rotor_npz['t'].T
# q_rotor = np.empty(0)
# regions_2d_rotor = motor_rotor_npz['regions_2d']
# regions_1d_rotor = motor_rotor_npz['regions_1d']

# motor_npz = np.load('../meshes/motor_pizza.npz', allow_pickle = True)

# geoOCC = motor_npz['geoOCC'].tolist()
# m = motor_npz['m']; m_new = m
# j3 = motor_npz['j3']

# import ngsolve as ng
# geoOCCmesh = geoOCC.GenerateMesh()
# ngsolve_mesh = ng.Mesh(geoOCCmesh)
# ngsolve_mesh.Refine()
# ngsolve_mesh.Refine()
# ngsolve_mesh.Refine()
# ngsolve_mesh.Refine()
# MESH = pde.mesh.netgen(ngsolve_mesh.ngmesh)

linear = '*air,*magnet,shaft_iron,*coil'
nonlinear = 'stator_iron,rotor_iron'
rotor = 'rotor_iron,*magnet,rotor_air,shaft_iron'


motor_npz = np.load('../meshes/data.npz', allow_pickle = True)
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

level = 1

# MESH = pde.mesh.netgen(ngsolvemesh.ngmesh)
open_file = open('mesh'+str(level)+'.pkl', "rb")
MESH = dill.load(open_file)[0]
open_file.close()

print(MESH)

##########################################################################################



##########################################################################################
# Extract indices
##########################################################################################

# R = lambda x: np.array([[np.cos(x),-np.sin(x)],
#                         [np.sin(x), np.cos(x)]])

# r1 = p[edges_rotor_outer[0,0],0]
# a1 = 2*np.pi/edges_rotor_outer.shape[0]

# Adjust points on the outer rotor to be equally spaced.
# for k in range(edges_rotor_outer.shape[0]):
#     p[edges_rotor_outer[k,0],0] = r1*np.cos(a1*(k))
#     p[edges_rotor_outer[k,0],1] = r1*np.sin(a1*(k))
##########################################################################################




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

phix_Hcurl = phi_Hcurl(int_order)[0];
phiy_Hcurl = phi_Hcurl(int_order)[1];


C = phi_L2(int_order) @ D(int_order) @ curlphi_Hcurl(int_order).T
Z = sps.csc_matrix((C.shape[0],C.shape[0]))


fem_linear = pde.int.evaluate(MESH, order = int_order, regions = linear).diagonal()
fem_nonlinear = pde.int.evaluate(MESH, order = int_order, regions = nonlinear).diagonal()
fem_rotor = pde.int.evaluate(MESH, order = int_order, regions = rotor).diagonal()
fem_air_gap_rotor = pde.int.evaluate(MESH, order = int_order, regions = 'air_gap_rotor').diagonal()

Ja = 0; J0 = 0
for i in range(48):
    Ja += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : j3[i], regions ='coil'+str(i+1)).diagonal()
    J0 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : j3[i], regions = 'coil'+str(i+1)).diagonal()
Ja = 0*Ja
J0 = 0*J0

M0 = 0; M1 = 0; M00 = 0; M10 = 0
for i in range(16):
    M0 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
    M1 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
    
    M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
    M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()

aM = phix_Hcurl@ D(int_order) @(M0) +\
     phiy_Hcurl@ D(int_order) @(M1)

aJ = phi_L2(int_order)@ D(int_order) @Ja

# iMh = pde.tools.fastBlockInverse(Mh1)
# S = C@iMh@C.T
# r = C@(iMh@aM)


# tm = time.monotonic(); x = sps.linalg.spsolve(S,r); print('dual: ',time.monotonic()-tm)
# MESH.pdesurf2(x)


from scipy.sparse import bmat

SYS = bmat([[Mh2,C.T],\
            [C,None]]).tocsc()

rhs = np.r_[aM,np.zeros(MESH.nt)]

# tm = time.monotonic(); x2 = sps.linalg.spsolve(SYS,rhs); print('mixed: ',time.monotonic()-tm)
# y2 = x2[MESH.NoEdges:]
# MESH.pdesurf2(y2)


##########################################################################################

spaceVh = 'N0d'

phix_d_Hcurl,phiy_d_Hcurl = pde.hcurl.assemble(MESH, space = spaceVh, matrix = 'phi', order = 4)
curlphi_d_Hcurl = pde.hcurl.assemble(MESH, space = spaceVh, matrix = 'curlphi', order = 4)

Md = phix_d_Hcurl @ D4 @ phix_d_Hcurl.T +\
     phiy_d_Hcurl @ D4 @ phiy_d_Hcurl.T

Cd = phi_L2(4) @ D4 @ curlphi_d_Hcurl.T

# spaceVh = 'N0d'
R0,R1,R2 = pde.hcurl.assembleE(MESH, space = spaceVh, matrix = 'M', order = 2)

phi_e = pde.l2.assembleE(MESH, space = 'P0', matrix = 'M', order = 2)

De = pde.int.assembleE(MESH, order = 1)
KK = phi_e @ De @ (R0+R1+R2).T

KK.data = KK.data*(np.abs(KK.data)>1e-13)
KK.eliminate_zeros()
KK

# KK = KK[np.r_[2*MESH.NonSingle_Edges,
#               2*MESH.NonSingle_Edges+1],:]

KK = KK[MESH.NonSingle_Edges,:]

sH = phix_d_Hcurl.shape[0]
sA = phi_L2_o1.shape[0]
sL = KK.shape[0]

aMd = phix_d_Hcurl@ D4 @(M0) +\
      phiy_d_Hcurl@ D4 @(M1)

SYS2 = bmat([[Md,Cd.T,KK.T],\
             [Cd,None,None],
             [KK,None,None]]).tocsc()

rhs2 = np.r_[aMd,np.zeros(sA+sL)]

# tm = time.monotonic(); x3 = sps.linalg.spsolve(SYS2,rhs2); print('mixed with decoupling: ',time.monotonic()-tm)
# H = x3[:sH]
# A = x3[sH:sH+sA]
# L = x3[sH+sA:]

##########################################################################################

# phix_d_Hcurl,phiy_d_Hcurl = pde.hcurl.assemble(MESH, space = spaceVh, matrix = 'phi', order = 1)

# Hx = phix_d_Hcurl.T@H; Hy = phiy_d_Hcurl.T@H

# fig = MESH.pdesurf_hybrid(dict(trig = 'P0', quad = 'Q0',controls = 1), A, u_height = 0)
# fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1), Hx**2+Hy**2, u_height = 0)
# fig.show()

##########################################################################################

iMd = pde.tools.fastBlockInverse(Md)
iBBd = pde.tools.fastBlockInverse(Cd@iMd@Cd.T)


SYS3 = -KK@iMd@KK.T + KK@iMd@Cd.T@iBBd@Cd@iMd@KK.T
rhs3 = -KK@iMd@aMd + KK@iMd@Cd.T@iBBd@Cd@iMd@aMd


tm = time.monotonic(); x4 = sps.linalg.spsolve(SYS3,rhs3); print('reduced hybrid stuff: ',time.monotonic()-tm)
tm = time.monotonic(); x4 = chol(-SYS3).solve_A(-rhs3); print('reduced hybrid stuff2: ',time.monotonic()-tm)

print(SYS3.shape,SYS3.nnz)

tm = time.monotonic();
for i in range(100):
    x4 = chol(-SYS3).solve_A(-rhs3)
print('reduced hybrid stuff: ',time.monotonic()-tm)

# lam4 = x4
# y4 = iBBd@Cd@iMd@(aMd-KK.T@lam4)
# u4 = iMd@(-Cd.T@y4-KK.T@lam4+aMd)


# # print(np.linalg.norm(lam3-lam4,np.inf))
# # print(np.linalg.norm(y3-y4,np.inf))
# # print(np.linalg.norm(u3-u4,np.inf))

# MESH.pdesurf2(y4)
