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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
# cmap = matplotlib.colors.ListedColormap("limegreen")
cmap = plt.cm.jet
import dill
import pickle

metadata = dict(title = 'Motor')
writer = FFMpegWriter(fps = 50, metadata = metadata)

##########################################################################################
# Loading mesh
##########################################################################################

# motor_npz = np.load('../meshes/motor_pizza_gap.npz', allow_pickle = True)

# geoOCC = motor_npz['geoOCC'].tolist()
# m = motor_npz['m']; m_new = m
# j3 = motor_npz['j3']

# import ngsolve as ng
# geoOCCmesh = geoOCC.GenerateMesh()
# ngsolve_mesh = ng.Mesh(geoOCCmesh)

motor_npz = np.load('../meshes/data.npz', allow_pickle = True)
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

plot = 0
level = 2

open_file = open('mesh_full'+str(level)+'.pkl', "rb")
# open_file = open('mesh'+str(level)+'.pkl', "rb")
MESH = dill.load(open_file)[0]
open_file.close()

linear = '*air,*magnet,shaft_iron,*coil'
nonlinear = 'stator_iron,rotor_iron'
rotor = 'rotor_iron,*magnet,rotor_air,shaft_iron'
##########################################################################################

from findPoints import *

# getPoints(MESH)
# makeIdentifications(MESH)

rot_speed = 1
rots = 1

tor = np.zeros(rots)
energy = np.zeros(rots)

for k in range(rots):
    
    print('Step : ', k)

    ##########################################################################################
    # Assembling stuff
    ##########################################################################################
    
    space_Vh = 'N0'
    space_Qh = 'P0'
    int_order = 0
    
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
    # Ja = 0*Ja; J0 = 0*J0
    
    M0 = 0; M1 = 0; M00 = 0; M10 = 0
    for i in range(16):
        M0 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
        M1 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
        
        M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
        M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
    
    aM = phix_Hcurl@ D(int_order) @(M0) +\
         phiy_Hcurl@ D(int_order) @(M1)
         
    aM = aM
    
    aJ = phi_L2(int_order)@ D(int_order) @Ja
    
    ##########################################################################################
    # RS = getRS_Hcurl(MESH, 1, 'N0', k, rot_speed)
    RS = sps.eye(MESH.NoEdges)
    ##########################################################################################
    
    SYS = bmat([[Mh2,C.T],\
                [C,None]]).tocsc()
    rhs = np.r_[aM,np.zeros(MESH.nt)]
    
    SYS2= bmat([[RS@Mh2@RS.T,RS@C.T],\
                [C@RS.T,None]]).tocsc()
    
    rhs2= np.r_[RS@aM,aJ+np.zeros(MESH.nt)]
    
    # tm = time.monotonic(); x = sps.linalg.spsolve(SYS,rhs); print('mixed: ',time.monotonic()-tm)
    tm = time.monotonic(); x2 = sps.linalg.spsolve(SYS2,rhs2); print('mixed: ',time.monotonic()-tm)
    
    
    A = x2[-MESH.nt:]
    H = RS.T@x2[:-MESH.nt]
    
    ##########################################################################################
    
    Hx = phix_Hcurl.T@H; Hy = phiy_Hcurl.T@H
    
    if plot == 1:
        if k == 0:
            fig = plt.figure()
            writer.setup(fig, "writer_test.mp4", 500)
            fig.show()
            ax1 = fig.add_subplot(111)
            # ax1 = fig.add_subplot(211)
            # ax2 = fig.add_subplot(212)
        
        tm = time.monotonic()
        ax1.cla()
        ax1.set_aspect(aspect = 'equal')
        MESH.pdesurf2(A, ax = ax1)
        # MESH.pdemesh2(ax = ax)
        MESH.pdegeom(ax = ax1)
        Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
        # ax1.tricontour(Triang, u[:MESH.np], levels = 25, colors = 'k', linewidths = 0.5, linestyles = 'solid')
    
    # ax2.cla()
    # ax2.set_aspect(aspect = 'equal')
    # MESH.pdesurf2(Hx**2+Hy**2, ax = ax2)
    # # MESH.pdemesh2(ax = ax)
    # MESH.pdegeom(ax = ax2)
    # Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
    
        writer.grab_frame()
    
    ##########################################################################################
    
    if rots > 1:
        rotor = 'rotor_iron,*magnet,rotor_air,shaft_iron,air_gap_rotor'
        
        fem_rotor = pde.int.evaluate(MESH, order = 0, regions = rotor).diagonal()
        trig_rotor = MESH.t[np.where(fem_rotor)[0],0:3]
        points_rotor = np.unique(trig_rotor)
        
        R = lambda x: np.array([[np.cos(x),-np.sin(x)],
                                [np.sin(x), np.cos(x)]])
        
        a1 = 2*np.pi/ident_edges_gap.shape[0]/8
        
        p_new = MESH.p.copy(); t_new = MESH.t.copy()
        p_new[points_rotor,:] = (R(a1*rot_speed)@MESH.p[points_rotor,:].T).T
        
        m_new = R(a1*rot_speed)@m_new
        
        MESH = pde.mesh(p_new,MESH.e,MESH.t,np.empty(0),MESH.regions_2d,MESH.regions_1d)
        # MESH.p[points_rotor,:] = (R(a1*rt)@MESH.p[points_rotor,:].T).T
    ##########################################################################################

if plot == 1:    
    writer.finish()
