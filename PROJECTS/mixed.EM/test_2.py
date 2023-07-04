import sys
sys.path.insert(0,'../../') # adds parent directory

import numpy as npy
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

nu0 = 10**7/(4*np.pi)
MESH = pde.mesh(p,e,t,q)

def spaceInfo(MESH,space):
    
    LISTS = MESH.FEMLISTS
    
    ###########################################################################
    if space == 'N0':
        
        LISTS['N0'] = {}
        LISTS['N0']['TRIG'] = {}
        
        LISTS['N0']['TRIG']['qp_we_M'] = pde.quadrature.dunavant(order = 2)
        LISTS['N0']['TRIG']['qp_we_Mh'] = pde.quadrature.dunavant(order = 1)
        LISTS['N0']['TRIG']['qp_we_K'] = pde.quadrature.dunavant(order = 0)
        
        LISTS['N0']['TRIG']['sizeM'] = MESH.NoEdges
        
        LISTS['N0']['TRIG']['phi'] = {}
        LISTS['N0']['TRIG']['phi'][0] = lambda x,y: np.r_[-y,x]
        LISTS['N0']['TRIG']['phi'][1] = lambda x,y: np.r_[-y,x-1]
        LISTS['N0']['TRIG']['phi'][2] = lambda x,y: np.r_[-y+1,x]
        
        LISTS['N0']['TRIG']['curlphi'] = {}
        LISTS['N0']['TRIG']['curlphi'][0] = lambda x,y: 2+0*x
        LISTS['N0']['TRIG']['curlphi'][1] = lambda x,y: 2+0*x
        LISTS['N0']['TRIG']['curlphi'][2] = lambda x,y: 2+0*x
        
        LISTS['N0']['TRIG']['LIST_DOF'] = MESH.TriangleToEdges
        LISTS['N0']['TRIG']['DIRECTION_DOF'] = MESH.EdgeDirectionTrig

        LISTS['N0']['B'] = {}
        LISTS['N0']['B']['phi'] = {}
        LISTS['N0']['B']['phi'][0] = lambda x: 1
        LISTS['N0']['B']['qp_we_B'] = pde.quadrature.one_d(order = 0)
        
        LISTS['N0']['B']['LIST_DOF'] = MESH.Boundary_Edges
        LISTS['N0']['B']['LIST_DOF_E1'] = MESH.IntEdgesToTriangles[:,0]
        LISTS['N0']['B']['LIST_DOF_E1'] = MESH.IntEdgesToTriangles[:,1]
        
        LISTS['N0']['TRIG']['phidual'] = {}
        LISTS['N0']['TRIG']['phidual'][0] = lambda x: 2+0*x
    ###########################################################################
    if space == 'NC1':
        
        LISTS['NC1'] = {}
        LISTS['NC1']['TRIG'] = {}
        
        LISTS['NC1']['TRIG']['qp_we_M'] = pde.quadrature.dunavant(order = 2)
        LISTS['NC1']['TRIG']['qp_we_Mh'] = pde.quadrature.dunavant(order = 1)
        LISTS['NC1']['TRIG']['qp_we_K'] = pde.quadrature.dunavant(order = 0)
        
        LISTS['NC1']['TRIG']['sizeM'] = 2*MESH.NoEdges
        
        LISTS['NC1']['TRIG']['phi'] = {}
        LISTS['NC1']['TRIG']['phi'][0] = lambda x,y: np.r_[0*x,x]
        LISTS['NC1']['TRIG']['phi'][1] = lambda x,y: np.r_[-y,0*x]
        LISTS['NC1']['TRIG']['phi'][2] = lambda x,y: np.r_[-y,-y]
        LISTS['NC1']['TRIG']['phi'][3] = lambda x,y: np.r_[0*x,x+y-1]
        LISTS['NC1']['TRIG']['phi'][4] = lambda x,y: np.r_[-x-y+1,0*y]
        LISTS['NC1']['TRIG']['phi'][5] = lambda x,y: np.r_[x,x]
        
        LISTS['NC1']['TRIG']['curlphi'] = {}
        LISTS['NC1']['TRIG']['curlphi'][0] = lambda x,y: 1+0*x
        LISTS['NC1']['TRIG']['curlphi'][1] = lambda x,y: 1+0*x
        LISTS['NC1']['TRIG']['curlphi'][2] = lambda x,y: 1+0*x
        LISTS['NC1']['TRIG']['curlphi'][3] = lambda x,y: 1+0*x
        LISTS['NC1']['TRIG']['curlphi'][4] = lambda x,y: 1+0*x
        LISTS['NC1']['TRIG']['curlphi'][5] = lambda x,y: 1+0*x
        
        LISTS['NC1']['TRIG']['LIST_DOF'] = np.c_[2*MESH.TriangleToEdges[:,0]   -1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                 2*MESH.TriangleToEdges[:,0]+1 +1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                 2*MESH.TriangleToEdges[:,1]   -1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                 2*MESH.TriangleToEdges[:,1]+1 +1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                 2*MESH.TriangleToEdges[:,2]   -1/2*(MESH.EdgeDirectionTrig[:,2]-1),
                                                 2*MESH.TriangleToEdges[:,2]+1 +1/2*(MESH.EdgeDirectionTrig[:,2]-1)]
    
        LISTS['NC1']['TRIG']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTrig[:,0],
                                                      MESH.EdgeDirectionTrig[:,0],
                                                      MESH.EdgeDirectionTrig[:,1],
                                                      MESH.EdgeDirectionTrig[:,1],
                                                      MESH.EdgeDirectionTrig[:,2],
                                                      MESH.EdgeDirectionTrig[:,2]]
        
        LISTS['NC1']['B'] = {}
        LISTS['NC1']['B']['phi'] = {}
        LISTS['NC1']['B']['phi'][0] = lambda x: 1-x
        LISTS['NC1']['B']['phi'][1] = lambda x: x
        LISTS['NC1']['B']['qp_we_B'] = pde.quadrature.one_d(order = 2)
        
        LISTS['NC1']['B']['LIST_DOF'] = np.c_[2*MESH.Boundary_Edges,
                                              2*MESH.Boundary_Edges + 1]
        
        LISTS['NC1']['B']['LIST_DOF_E1'] = np.c_[2*MESH.IntEdgesToTriangles[:,0],
                                                 2*MESH.IntEdgesToTriangles[:,0]+1]
        LISTS['NC1']['B']['LIST_DOF_E1'] = np.c_[2*MESH.IntEdgesToTriangles[:,1],
                                                 2*MESH.IntEdgesToTriangles[:,1]+1]

                                              
        LISTS['NC1']['B']['LIST_DOF_E'] = np.c_[2*MESH.NonSingle_Edges,
                                                2*MESH.NonSingle_Edges + 1]
        
        LISTS['NC1']['TRIG']['phidual'] = {}
        LISTS['NC1']['TRIG']['phidual'][0] = lambda x: 6*x-2 # dual to x and 1-x on the edge...
        LISTS['NC1']['TRIG']['phidual'][1] = lambda x: -6*x+4
    ###########################################################################


space = 'NC1'
edges = npy.empty(0)
matrix = 'M'
order = 3

if not space in MESH.FEMLISTS.keys():
    spaceInfo(MESH,space)

if not hasattr(MESH, 'Boundary_EdgeOrientation'):
    MESH.makeBEO()

p = MESH.p
e = MESH.e; ne = e.shape[0]

phi = MESH.FEMLISTS[space]['B']['phi']; lphi = len(phi)

if edges.size == 0:
    e = MESH.e
else:
    e = MESH.EdgesToVertices[edges,:]
    
ne = e.shape[0]
LIST_DOF = MESH.FEMLISTS[space]['B']['LIST_DOF']

if order != -1:
    qp,we = pde.quadrature.one_d(order); nqp = len(we)
        
#####################################################################################
# Mappings
#####################################################################################
    
e0 = e[:,0]; e1 = e[:,1]
A0 = p[e1,0]-p[e0,0]; A1 = p[e1,1]-p[e0,1]
detA = npy.sqrt(A0**2+A1**2)

if edges.size == 0:
    detA = detA*MESH.Boundary_EdgeOrientation

#####################################################################################
# Mass matrix (over the edge)
#####################################################################################

LIST_DOF2 = np.r_[0:ne]

if matrix == 'M':
    if order == -1:
        qp = MESH.FEMLISTS[space]['B']['qp_we_B'][0]; 
        we = MESH.FEMLISTS[space]['B']['qp_we_B'][1]; nqp = len(we)
    
    ellmatsB = npy.zeros((nqp*ne,lphi))
    
    im = npy.tile(LIST_DOF,(nqp,1))
    jm = npy.tile(npy.tile(LIST_DOF2*nqp,nqp) + npy.arange(nqp).repeat(ne),lphi)
    
    # jm = npy.tile(npy.c_[0:ne*nqp].reshape(ne,nqp).T.flatten(),(lphi,1)).T
    
    for j in range(lphi):
        for i in range(nqp):
            ellmatsB[i*ne:(i+1)*ne,j] = 1/npy.abs(detA)*phi[j](qp[i])
            
            
            
            
            
aa = npy.c_[0:ne*nqp].reshape(ne,nqp).T.flatten()

bb = np.tile(np.tile(np.arange(ne)*nqp,nqp) + np.arange(nqp).repeat(ne),lphi)



    