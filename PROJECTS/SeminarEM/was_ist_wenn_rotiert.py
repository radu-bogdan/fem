import sys
sys.path.insert(0,'../../') # adds parent directory
# sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
from sksparse.cholmod import cholesky as chol
import plotly.io as pio
pio.renderers.default = 'browser'
import matplotlib.pyplot as plt
import time
import numba as nb


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
m = motor_npz['m']
j3 = motor_npz['j3']

##########################################################################################

ind_rotor_outer = np.flatnonzero(np.core.defchararray.find(list(regions_1d),'rotor_outer')!=-1)
ind_edges_rotor_outer = np.where(np.isin(e[:,2],ind_rotor_outer))[0]


edges_rotor_outer = e[ind_edges_rotor_outer,0:2]

r1 = p[edges_rotor_outer[0,0],0]
a1 = 2*np.pi/edges_rotor_outer.shape[0]

alpha = np.empty(edges_rotor_outer.shape[0])
radius = np.empty(edges_rotor_outer.shape[0])


# Adjust points on the outer rotor to be equally spaced.
for k in range(edges_rotor_outer.shape[0]):
    p[edges_rotor_outer[k,0],0] = r1*np.cos(a1*(k))
    p[edges_rotor_outer[k,0],1] = r1*np.sin(a1*(k))



# Check angles, radii
for i in range(edges_rotor_outer.shape[0]):
    a = p[edges_rotor_outer[i,0],:]
    b = p[edges_rotor_outer[i,1],:]
    l = np.linalg.norm(a-b,2)
    
    alpha[i] = 2*np.arcsin(l/(2*r1))
    radius[i] = np.linalg.norm(a,2)
    


# Find all the points in the rotor
def getIndices(liste, name, exact = 0, return_index = False):
    if exact == 0:
        ind = np.flatnonzero(np.core.defchararray.find(list(liste),name)!=-1)
    else:
        ind = [i for i, x in enumerate(list(liste)) if x == name]
    elem = np.where(np.isin(t[:,3],ind))[0]
    mask = np.zeros(t.shape[0]); mask[elem] = 1
    if return_index:
        return ind, mask
    else:
        return mask

ind_iron_rotor, mask_iron_rotor = getIndices(regions_2d, 'rotor_iron', exact = 1, return_index = True)
ind_rotor_air, mask_rotor_air = getIndices(regions_2d, 'rotor_air', exact = 1, return_index = True)
ind_air_gap_rotor, mask_air_gap_rotor = getIndices(regions_2d, 'air_gap_rotor', exact = 1, return_index = True)

# mask_air_all = getIndices(regions_2d, 'air')
# mask_stator_rotor_and_shaft = getIndices(regions_2d, 'iron')
ind_magnet, mask_magnet = getIndices(regions_2d, 'magnet', return_index = True)
# mask_coil = getIndices(regions_2d, 'coil')
mask_shaft = getIndices(regions_2d, 'shaft')

# mask_linear    = mask_air_all + mask_magnet + mask_shaft + mask_coil
# mask_nonlinear = mask_stator_rotor_and_shaft - mask_shaft

mask_rotor  = mask_iron_rotor + mask_magnet + mask_rotor_air + mask_shaft
trig_rotor = t[np.where(mask_rotor)[0],0:3]
trig_air_gap_rotor = t[np.where(mask_air_gap_rotor)[0],0:3]
points_rotor = np.unique(trig_rotor)

trig_air_gap_rotor_old = trig_air_gap_rotor.copy()
trig_air_gap_rotor_new = trig_air_gap_rotor.copy()




R = lambda x: np.array([[np.cos(x),-np.sin(x)],
                        [np.sin(x), np.cos(x)]])

rt = 10

# @profile
# def do():

@nb.njit()
def dojit(t, edges_rotor_outer, trig_air_gap_rotor, shifted_coeff):
    trig_air_gap_rotor_new = trig_air_gap_rotor.copy()
    for k in range(edges_rotor_outer.shape[0]):
        a = np.where(trig_air_gap_rotor == edges_rotor_outer[k,0])
        for j in range(a[0].shape[0]):
            trig_air_gap_rotor_new[a[0][j],a[1][j]] = shifted_coeff[k]
    return trig_air_gap_rotor_new

def ismember(a_vec, b_vec):
    """ MATLAB equivalent ismember function """
    bool_ind = np.isin(a_vec,b_vec)
    common = a_vec[bool_ind]
    common_unique, common_inv  = np.unique(common, return_inverse=True)     # common = common_unique[common_inv]
    b_unique, b_ind = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    return bool_ind, common_ind[common_inv]

MESH = pde.mesh(p,e,t,q)

tm = time.monotonic()

shifted_coeff = np.roll(edges_rotor_outer[:,0],rt)
kk, jj = MESH._mesh__ismember(trig_air_gap_rotor_new,edges_rotor_outer[:,0])
trig_air_gap_rotor_new[kk] = shifted_coeff[jj]

p_new = p.copy(); t_new = t.copy()
p_new[points_rotor,:] = (R(a1*rt)@p[points_rotor,:].T).T
t_new[np.where(mask_air_gap_rotor)[0],0:3] = trig_air_gap_rotor_new

print('Solving took ', time.monotonic()-tm, 'seconds')

MESH2 = pde.mesh(p_new,e,t_new,q)

# aa = np.zeros(MESH.np)
# aa[points_rotor] = 1

# MP = pde.int.evaluate(MESH, order = 0, regions = np.sort(np.r_[ind_iron_rotor,ind_rotor_air,ind_magnet])).diagonal()
# fig = MESH.pdesurf_hybrid(dict(trig = 'P1', quad = 'Q0', controls = 1), aa, u_height = 0)
# fig.show()

# MP = pde.int.evaluate(MESH2, order = 0, regions = np.sort(np.r_[ind_iron_rotor,ind_rotor_air,ind_magnet])).diagonal()
# fig = MESH2.pdesurf_hybrid(dict(trig = 'P1', quad = 'Q0', controls = 1), aa, u_height = 0)
# fig.show()


