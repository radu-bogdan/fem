import sys
sys.path.insert(0,'../../') # adds parent directory

import pde
import numpy as np
import ngsolve as ng
import dill
import pickle

# from netgen import gui

motor_npz = np.load('../meshes/motor_pizza_gap.npz', allow_pickle = True)
# motor_npz = np.load('../meshes/motor.npz', allow_pickle = True)

geoOCC = motor_npz['geoOCC'].tolist()
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

geoOCCmesh = geoOCC.GenerateMesh()
ngsolvemesh = ng.Mesh(geoOCCmesh)

nums = 2
MESH = []

for i in range(nums):
    # ngsolvemesh.Refine()
    ngsolvemesh.ngmesh.Refine()

MESH.append(pde.mesh.netgen(ngsolvemesh.ngmesh))


open_file = open('mesh_full'+str(nums)+'.pkl', "wb")
dill.dump(MESH, open_file)
open_file.close()

m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

######################################################################
# Loading script: 
######################################################################

import sys
sys.path.insert(0,'../../') # adds parent directory
import dill
import pickle
import pde

nums = 2
open_file = open('mesh_full'+str(nums)+'.pkl', "rb")
MESH_LOADED = dill.load(open_file)
open_file.close()



print(MESH_LOADED[0])