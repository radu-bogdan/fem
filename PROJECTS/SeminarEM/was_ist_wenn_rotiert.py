import sys
sys.path.insert(0,'../../') # adds parent directory
# sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
from sksparse.cholmod import cholesky as chol
import plotly.io as pio
pio.renderers.default = 'browser'
import nonlinear_Algorithms
import numba as nb
import pyamg
import matplotlib.pyplot as plt

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
m = motor_npz['m']
j3 = motor_npz['j3']

##########################################################################################

ind_rotor_outer = np.flatnonzero(np.core.defchararray.find(list(regions_1d),'rotor_outer')!=-1)
ind_edges_rotor_outer = np.where(np.isin(e[:,2],ind_rotor_outer))[0]


edges_rotor_outer = e[ind_edges_rotor_outer,0:2]

r = p[edges_rotor_outer[0,0],0]

alpha = np.empty(edges_rotor_outer.shape[0])
for i in range(edges_rotor_outer.shape[0]):
    a = p[edges_rotor_outer[i,0],:]
    b = p[edges_rotor_outer[i,1],:]
    l = np.linalg.norm(a-b,2)
    
    alpha[i] = 2*np.arcsin(l/(2*r))

plt.plot(alpha)