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

import matplotlib.pyplot as plt
import matplotlib
cmap = plt.cm.jet

##########################################################################################
# Loading mesh
##########################################################################################
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
# MESH.refinemesh()
t = MESH.t
p = MESH.p
##########################################################################################



##########################################################################################
# Extract indices
##########################################################################################

mask_air_all = MESH.getIndices2d(regions_2d, 'air')
mask_stator_rotor_and_shaft = MESH.getIndices2d(regions_2d, 'iron')
mask_magnet = MESH.getIndices2d(regions_2d, 'magnet')
mask_coil = MESH.getIndices2d(regions_2d, 'coil')
mask_shaft = MESH.getIndices2d(regions_2d, 'shaft')
mask_iron_rotor = MESH.getIndices2d(regions_2d, 'rotor_iron', exact = 1)
mask_rotor_air = MESH.getIndices2d(regions_2d, 'rotor_air', exact = 1)
mask_air_gap_rotor = MESH.getIndices2d(regions_2d, 'air_gap_rotor', exact = 1)

mask_rotor     = mask_iron_rotor + mask_magnet + mask_rotor_air + mask_shaft
mask_linear    = mask_air_all + mask_magnet + mask_shaft + mask_coil
mask_nonlinear = mask_stator_rotor_and_shaft - mask_shaft

trig_rotor = MESH.t[np.where(mask_rotor)[0],0:3]
trig_air_gap_rotor = MESH.t[np.where(mask_air_gap_rotor)[0],0:3]
points_rotor = np.unique(trig_rotor)
points_rotor_and_airgaprotor = np.unique(np.r_[trig_rotor,trig_air_gap_rotor])


ind_stator_outer = np.flatnonzero(np.core.defchararray.find(list(regions_1d),'stator_outer')!=-1)
ind_rotor_outer = np.flatnonzero(np.core.defchararray.find(list(regions_1d),'rotor_outer')!=-1)
ind_edges_rotor_outer = np.where(np.isin(e[:,2],ind_rotor_outer))[0]
edges_rotor_outer = e[ind_edges_rotor_outer,0:2]

ind_trig_coils   = MESH.getIndices2d(regions_2d, 'coil', return_index = True)[0]
ind_trig_magnets = MESH.getIndices2d(regions_2d, 'magnet', return_index = True)[0]

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
phi_Hdiv = lambda x : pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'phi', order = x)   
curlphi_Hcurl = lambda x : pde.hcurl.assemble(MESH, space = 'NC1', matrix = 'curlphi', order = x)
phi_L2 = lambda x : pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = x)

D = lambda x : pde.int.assemble(MESH, order = x)

Mh = lambda x: phi_Hcurl(x)[0] @ D(x) @ phi_Hcurl(x)[0].T + \
               phi_Hcurl(x)[1] @ D(x) @ phi_Hcurl(x)[1].T
     
mixedMh = lambda x: phi_Hcurl(x)[0] @ D(x) @ phi_Hdiv(x)[0].T + \
                    phi_Hcurl(x)[1] @ D(x) @ phi_Hdiv(x)[1].T
                    

D1 = D(1); D2 = D(2); Mh1 = Mh(1)
phi_L2_o1 = phi_L2(1)
curlphi_Hcurl_o1 = curlphi_Hcurl(1)

phix_Hcurl_o1 = phi_Hcurl(1)[0];
phiy_Hcurl_o1 = phi_Hcurl(1)[1];

C = phi_L2_o1 @ D1 @ curlphi_Hcurl_o1.T

iMh = pde.tools.fastBlockInverse(Mh1)
S = C@iMh@C.T


M0 = 0; M1 = 0; M00 = 0; M10 = 0
for i in range(16):
    M0 += pde.int.evaluate(MESH, order = 1, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    M1 += pde.int.evaluate(MESH, order = 1, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    
    M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()

aM = phix_Hcurl_o1@ D1 @(M0) +\
     phiy_Hcurl_o1@ D1 @(M1)

r = C@(iMh@aM)

x = sps.linalg.spsolve(S,r)




dphix_H1_o1, dphiy_H1_o1 = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 1)

Kxx = dphix_H1_o1 @ D1 @ dphix_H1_o1.T
Kyy = dphiy_H1_o1 @ D1 @ dphiy_H1_o1.T

phi_H1b = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = dphix_H1_o1.shape, order = 1)

D_stator_outer = pde.int.evaluateB(MESH, order = 1, edges = ind_stator_outer)
B_stator_outer = phi_H1b@ D_stator_outer @ phi_H1b.T

aM = dphix_H1_o1@ D1 @(-M1) +\
     dphiy_H1_o1@ D1 @(+M0)
     
x2 = sps.linalg.spsolve(Kxx+Kyy+10**10*B_stator_outer,aM)
    
##########################################################################################
# Plotting stuff
##########################################################################################


fig = plt.figure()
fig.show()
ax1 = fig.add_subplot(111)
ax1.set_aspect(aspect = 'equal')

ax1.cla()

MESH.pdegeom(ax = ax1)
# MESH.pdemesh2(ax = ax1)

Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
chip = ax1.tripcolor(Triang, x, cmap = cmap, shading = 'flat', lw = 0.1)
    
    
chip = ax1.tripcolor(Triang, x2, cmap = cmap, shading = 'gouraud', lw = 0.1)
