import sys
sys.path.insert(0,'../../') # adds parent directory
# sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
# from sksparse.cholmod import cholesky as chol
import plotly.io as pio
pio.renderers.default = 'browser'
import nonlinear_Algorithms
import numba as nb
import pyamg

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
# cmap = matplotlib.colors.ListedColormap("limegreen")
cmap = plt.cm.jet
# ax2.set_aspect(aspect = 'equal')

# cbar = plt.colorbar(ax)

# @profile
# def do():
    
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

MESH = pde.mesh(p,e,t,q)

ind_air_all = np.flatnonzero(np.core.defchararray.find(list(regions_2d),'air')!=-1)
ind_stator_rotor = np.flatnonzero(np.core.defchararray.find(list(regions_2d),'iron')!=-1)
ind_magnet = np.flatnonzero(np.core.defchararray.find(list(regions_2d),'magnet')!=-1)
ind_coil = np.flatnonzero(np.core.defchararray.find(list(regions_2d),'coil')!=-1)
ind_shaft = np.flatnonzero(np.core.defchararray.find(list(regions_2d),'shaft')!=-1)

trig_air_all = np.where(np.isin(t[:,3],ind_air_all))
trig_stator_rotor = np.where(np.isin(t[:,3],ind_stator_rotor))
trig_magnet = np.where(np.isin(t[:,3],ind_magnet))
trig_coil = np.where(np.isin(t[:,3],ind_coil))
trig_shaft = np.where(np.isin(t[:,3],ind_shaft))

vek = np.zeros(MESH.nt)
vek[trig_air_all] = 1
vek[trig_magnet] = 2
vek[trig_coil] = 3
vek[trig_stator_rotor] = 4
vek[trig_shaft] = np.nan

# fig = MESH.pdemesh()
# fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1), vek, u_height=0)
# fig.show()


fig = plt.figure()
fig.show()
ax1 = fig.add_subplot(111)
ax1.set_aspect(aspect = 'equal')
cmap = plt.cm.rainbow
Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
chip = ax1.tripcolor(Triang, vek, cmap = cmap, shading = 'flat', lw = 0.1)
plt.axis('off')
newName = 'kek'
plt.savefig(newName + '.png', bbox_inches='tight', pad_inches=-0.15, dpi=500)