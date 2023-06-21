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
import pyamg
from scipy.sparse import hstack,vstack


import matplotlib.pyplot as plt
import matplotlib
cmap = plt.cm.jet

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

motor_npz = np.load('meshes/motor.npz', allow_pickle = True)

p = motor_npz['p'].T
e = motor_npz['e'].T
t = motor_npz['t'].T
q = np.empty(0)
regions_2d = motor_npz['regions_2d']
regions_1d = motor_npz['regions_1d']
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']
##########################################################################################



##########################################################################################
# Parameters
##########################################################################################

nu0 = 10**7/(4*np.pi)

MESH = pde.mesh(p,e,t,q)
# MESH.refinemesh()
# MESH.refinemesh()

##########################################################################################



##########################################################################################
# Extract indices
##########################################################################################

ind_stator_outer = MESH.getIndices2d(regions_1d, 'stator_outer', return_index = True)[0]
ind_rotor_outer = MESH.getIndices2d(regions_1d, 'rotor_outer', return_index = True)[0]

ind_edges_rotor_outer = np.where(np.isin(e[:,2],ind_rotor_outer))[0]
edges_rotor_outer = e[ind_edges_rotor_outer,0:2]

ind_air_gap_rotor = MESH.getIndices2d(regions_2d, 'air_gap_rotor', return_index = True)[0]
ind_trig_coils = MESH.getIndices2d(regions_2d, 'coil', return_index = True)[0]
ind_trig_magnets = MESH.getIndices2d(regions_2d, 'magnet', return_index = True)[0]
ind_air_all = MESH.getIndices2d(regions_2d, 'air', return_index = True)[0]
ind_magnet = MESH.getIndices2d(regions_2d, 'magnet', return_index = True)[0]
ind_shaft = MESH.getIndices2d(regions_2d, 'shaft', return_index = True)[0]
ind_coil = MESH.getIndices2d(regions_2d, 'coil', return_index = True)[0]
ind_stator_rotor_and_shaft = MESH.getIndices2d(regions_2d, 'iron', return_index = True)[0]
ind_iron_rotor = MESH.getIndices2d(regions_2d, 'rotor_iron', return_index = True)[0]
ind_rotor_air = MESH.getIndices2d(regions_2d, 'rotor_air', return_index = True)[0]

ind_linear = np.r_[ind_air_all,ind_magnet,ind_shaft,ind_coil]
ind_nonlinear = np.setdiff1d(ind_stator_rotor_and_shaft,ind_shaft)
ind_rotor = np.r_[ind_iron_rotor,ind_magnet,ind_rotor_air,ind_shaft]

R = lambda x: np.array([[np.cos(x),-np.sin(x)],
                        [np.sin(x), np.cos(x)]])

r1 = p[edges_rotor_outer[0,0],0]
a1 = 2*np.pi/edges_rotor_outer.shape[0]

# Adjust points on the outer rotor to be equally spaced.
for k in range(edges_rotor_outer.shape[0]):
    p[edges_rotor_outer[k,0],0] = r1*np.cos(a1*(k))
    p[edges_rotor_outer[k,0],1] = r1*np.sin(a1*(k))
##########################################################################################




##########################################################################################
# Assembling stuff
##########################################################################################

# u = np.zeros(MESH.np)

tm = time.monotonic()

phi_Hcurl = lambda x : pde.hcurl.assemble(MESH, space = 'NC1', matrix = 'phi', order = x)
curlphi_Hcurl = lambda x : pde.hcurl.assemble(MESH, space = 'NC1', matrix = 'curlphi', order = x)
phi_L2 = lambda x : pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = x)

D = lambda x : pde.int.assemble(MESH, order = x)

Mh = lambda x: phi_Hcurl(x)[0] @ D(x) @ phi_Hcurl(x)[0].T + \
               phi_Hcurl(x)[1] @ D(x) @ phi_Hcurl(x)[1].T

D1 = D(1); D2 = D(2); D4 = D(4); Mh1 = Mh(1); Mh2 = Mh(2)
phi_L2_o1 = phi_L2(1)
curlphi_Hcurl_o1 = curlphi_Hcurl(1)

phix_Hcurl = phi_Hcurl(4)[0];
phiy_Hcurl = phi_Hcurl(4)[1];

C = phi_L2_o1 @ D1 @ curlphi_Hcurl_o1.T


fem_linear = pde.int.evaluate(MESH, order = 4, regions = ind_linear).diagonal()
fem_nonlinear = pde.int.evaluate(MESH, order = 4, regions = ind_nonlinear).diagonal()
fem_rotor = pde.int.evaluate(MESH, order = 4, regions = ind_rotor).diagonal()
fem_air_gap_rotor = pde.int.evaluate(MESH, order = 4, regions = ind_air_gap_rotor).diagonal()


M0 = 0; M1 = 0; M00 = 0; M10 = 0
for i in range(16):
    M0 += pde.int.evaluate(MESH, order = 4, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    M1 += pde.int.evaluate(MESH, order = 4, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    
    M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()

aM = phix_Hcurl@ D(4) @(M0) +\
     phiy_Hcurl@ D(4) @(M1)


iMh = pde.tools.fastBlockInverse(Mh1)
S = C@iMh@C.T
r = C@(iMh@aM)

tm = time.monotonic(); x = sps.linalg.spsolve(S,r); print('dual: ',time.monotonic()-tm)
##########################################################################################


##########################################################################################

from nonlinLaws import *

a = np.random.randint(100_000, size = 1_000).astype(float)
b = np.random.randint(100_000, size = 1_000).astype(float)
g_nonlinear_all(a,b)

# starting from an initial guess Hn, compute

# ux = dphix_H1.T@u; uy = dphiy_H1.T@u

H = 1+np.zeros(2*MESH.NoEdges)
Hx = phix_Hcurl.T@H; Hy = phiy_Hcurl.T@H

g_H,gx_H,gy_H,gxx_H,gxy_H,gyx_H,gyy_H = g_nonlinear_all(Hx,Hy)

gxx_H_Mxx = phix_Hcurl @ D4 @ sps.diags(gxx_H*fem_nonlinear + gxx_linear(Hx,Hy)*fem_linear)@ phix_Hcurl.T
gyy_H_Myy = phiy_Hcurl @ D4 @ sps.diags(gyy_H*fem_nonlinear + gyy_linear(Hx,Hy)*fem_linear)@ phiy_Hcurl.T
gxy_H_Mxy = phiy_Hcurl @ D4 @ sps.diags(gxy_H*fem_nonlinear + gxy_linear(Hx,Hy)*fem_linear)@ phix_Hcurl.T
gyx_H_Myx = phix_Hcurl @ D4 @ sps.diags(gyx_H*fem_nonlinear + gyx_linear(Hx,Hy)*fem_linear)@ phiy_Hcurl.T

M = gxx_H_Mxx + gyy_H_Myy + gxy_H_Mxy + gyx_H_Myx
Z = sps.csc_matrix((C.shape[0],C.shape[0]))
S = vstack((hstack((M,-C.T)),
            hstack((C,Z))))
S1 = hstack((C,Z))


# def gss(u):
#     ux = dphix_H1.T@u; uy = dphiy_H1.T@u
    
#     fxx_grad_u_Kxx = dphix_H1 @ D_order_dphidphi @ sps.diags(fxx_linear(ux,uy)*fem_linear + fxx_nonlinear(ux,uy)*fem_nonlinear)@ dphix_H1.T
#     fyy_grad_u_Kyy = dphiy_H1 @ D_order_dphidphi @ sps.diags(fyy_linear(ux,uy)*fem_linear + fyy_nonlinear(ux,uy)*fem_nonlinear)@ dphiy_H1.T
#     fxy_grad_u_Kxy = dphiy_H1 @ D_order_dphidphi @ sps.diags(fxy_linear(ux,uy)*fem_linear + fxy_nonlinear(ux,uy)*fem_nonlinear)@ dphix_H1.T
#     fyx_grad_u_Kyx = dphix_H1 @ D_order_dphidphi @ sps.diags(fyx_linear(ux,uy)*fem_linear + fyx_nonlinear(ux,uy)*fem_nonlinear)@ dphiy_H1.T
#     return (fxx_grad_u_Kxx + fyy_grad_u_Kyy + fxy_grad_u_Kxy + fyx_grad_u_Kyx) + penalty*B_stator_outer


















##########################################################################################

dphix_H1_o1, dphiy_H1_o1 = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 4)

Kxx = dphix_H1_o1 @ D(4) @ dphix_H1_o1.T
Kyy = dphiy_H1_o1 @ D(4) @ dphiy_H1_o1.T

phi_H1b = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = dphix_H1_o1.shape, order = 4)

D_stator_outer = pde.int.evaluateB(MESH, order = 4, edges = ind_stator_outer)
B_stator_outer = phi_H1b@ D_stator_outer @ phi_H1b.T

aM = dphix_H1_o1@ D(4) @(-M1) +\
     dphiy_H1_o1@ D(4) @(+M0)
     


tm = time.monotonic(); x2 = sps.linalg.spsolve(Kxx+Kyy+10**10*B_stator_outer,aM); print('primal: ',time.monotonic()-tm)
    
##########################################################################################
# Plotting stuff
##########################################################################################


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.set_aspect(aspect = 'equal')

ax1.cla()
MESH.pdegeom(ax = ax1)
# MESH.pdemesh2(ax = ax1)

Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
chip = ax1.tripcolor(Triang, x, cmap = cmap, shading = 'flat', lw = 0.1)

ax2 = fig.add_subplot(122)
ax2.set_aspect(aspect = 'equal')

ax2.cla()
MESH.pdegeom(ax = ax2)
chip = ax2.tripcolor(Triang, x2, cmap = cmap, shading = 'gouraud', lw = 0.1)

fig.tight_layout()
fig.show()

