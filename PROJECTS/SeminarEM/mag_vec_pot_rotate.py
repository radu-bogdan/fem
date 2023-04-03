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
import matplotlib.colors
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
cmap = matplotlib.colors.ListedColormap("limegreen")
cmap = plt.cm.jet


metadata = dict(title = 'Movie Test', artist = 'Matplotlib', comment = 'Movie support!')
writer = FFMpegWriter(fps = 10, metadata = metadata)

# plt.ion()
fig = plt.figure()
fig.show()
ax = fig.add_subplot()
tpc = ax.set_aspect(aspect = 'equal')

writer.setup(fig, "writer_test.mp4", 500)

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



##########################################################################################
# Parameters
##########################################################################################

ORDER = 1
total = 1

nu0 = 10**7/(4*np.pi)

MESH = pde.mesh(p,e,t,q)
# MESH.refinemesh()
# MESH.refinemesh()
# MESH.refinemesh()
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

trig_rotor = t[np.where(mask_rotor)[0],0:3]
trig_air_gap_rotor = t[np.where(mask_air_gap_rotor)[0],0:3]
points_rotor = np.unique(trig_rotor)


ind_stator_outer = np.flatnonzero(np.core.defchararray.find(list(regions_1d),'stator_outer')!=-1)
ind_rotor_outer = np.flatnonzero(np.core.defchararray.find(list(regions_1d),'rotor_outer')!=-1)
ind_edges_rotor_outer = np.where(np.isin(e[:,2],ind_rotor_outer))[0]
edges_rotor_outer = e[ind_edges_rotor_outer,0:2]

ind_trig_coils   = MESH.getIndices2d(regions_2d, 'coil', return_index = True)[0]
ind_trig_magnets = MESH.getIndices2d(regions_2d, 'magnet', return_index = True)[0]

# ind_trig_coils = {}
# for i in range(48):
#     ind_trig_coils[i] = getIndices2d(regions_2d, 'coil' + str(i+1), exact = 1, return_index = True)[0]

# ind_trig_magnets = {}
# for i in range(16):
#     ind_trig_magnets[i] = getIndices2d(regions_2d, 'magnet' + str(i+1), exact = 1, return_index = True)[0]

R = lambda x: np.array([[np.cos(x),-np.sin(x)],
                        [np.sin(x), np.cos(x)]])

r1 = p[edges_rotor_outer[0,0],0]
a1 = 2*np.pi/edges_rotor_outer.shape[0]

# Adjust points on the outer rotor to be equally spaced.
# for k in range(edges_rotor_outer.shape[0]):
#     p[edges_rotor_outer[k,0],0] = r1*np.cos(a1*(k-1))
#     p[edges_rotor_outer[k,0],1] = r1*np.sin(a1*(k-1))
##########################################################################################



##########################################################################################
# Order configuration
##########################################################################################
if ORDER == 1:
    poly = 'P1'
    dxpoly = 'P0'
    order_phiphi = 2
    order_dphidphi = 0
    new_mask_linear = mask_linear
    new_mask_nonlinear = mask_nonlinear
    u = np.zeros(MESH.np)
    
if ORDER == 2:
    poly = 'P2'
    dxpoly = 'P1'
    order_phiphi = 4
    order_dphidphi = 2
    new_mask_linear = np.tile(mask_linear,(3,1)).T.flatten()
    new_mask_nonlinear = np.tile(mask_nonlinear,(3,1)).T.flatten()
    u = np.zeros(MESH.np + MESH.NoEdges)
##########################################################################################



##########################################################################################
# Brauer/Nonlinear laws ... ?
##########################################################################################

k1 = 49.4; k2 = 1.46; k3 = 520.6
# k1 = 3.8; k2 = 2.17; k3 = 396.2

f_iron = lambda x,y : k1/(2*k2)*(np.exp(k2*(x**2+y**2))-1) + 1/2*k3*(x**2+y**2) # magnetic energy density in iron

nu = lambda x,y : k1*np.exp(k2*(x**2+y**2))+k3
nux = lambda x,y : 2*x*k1*k2*np.exp(k2*(x**2+y**2))
nuy = lambda x,y : 2*y*k1*k2*np.exp(k2*(x**2+y**2))

fx_iron = lambda x,y : nu(x,y)*x
fy_iron = lambda x,y : nu(x,y)*y
fxx_iron = lambda x,y : nu(x,y) + x*nux(x,y)
fxy_iron = lambda x,y : x*nuy(x,y)
fyx_iron = lambda x,y : y*nux(x,y)
fyy_iron = lambda x,y : nu(x,y) + y*nuy(x,y)

f_linear = lambda x,y : 1/2*nu0*(x**2+y**2)
fx_linear = lambda x,y : nu0*x
fy_linear = lambda x,y : nu0*y
fxx_linear = lambda x,y : nu0 + 0*x
fxy_linear = lambda x,y : x*0
fyx_linear = lambda x,y : y*0
fyy_linear = lambda x,y : nu0 + 0*y


f   = lambda ux,uy :   f_linear(ux,uy)*new_mask_linear +   f_iron(ux,uy)*new_mask_nonlinear
fx  = lambda ux,uy :  fx_linear(ux,uy)*new_mask_linear +  fx_iron(ux,uy)*new_mask_nonlinear
fy  = lambda ux,uy :  fy_linear(ux,uy)*new_mask_linear +  fy_iron(ux,uy)*new_mask_nonlinear
fxx = lambda ux,uy : fxx_linear(ux,uy)*new_mask_linear + fxx_iron(ux,uy)*new_mask_nonlinear
fxy = lambda ux,uy : fxy_linear(ux,uy)*new_mask_linear + fxy_iron(ux,uy)*new_mask_nonlinear
fyx = lambda ux,uy : fyx_linear(ux,uy)*new_mask_linear + fyx_iron(ux,uy)*new_mask_nonlinear
fyy = lambda ux,uy : fyy_linear(ux,uy)*new_mask_linear + fyy_iron(ux,uy)*new_mask_nonlinear
###########################################################################################

rot_speed = 1; rt = 0
rots = 200
tor = np.zeros(rots)

for k in range(rots):
    
    ##########################################################################################
    # Assembling stuff
    ##########################################################################################
    
    # u = np.zeros(MESH.np)
    
    tm = time.monotonic()
    
    phi_H1  = pde.h1.assemble(MESH, space = poly, matrix = 'M', order = order_phiphi)
    dphix_H1, dphiy_H1 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_dphidphi)
    dphix_H1_o0, dphiy_H1_o0 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = 0)
    dphix_H1_o1, dphiy_H1_o1 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = 1)
    dphix_H1_order_phiphi, dphiy_H1_order_phiphi = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_phiphi)
    phi_H1b = pde.h1.assembleB(MESH, space = poly, matrix = 'M', shape = phi_H1.shape, order = order_phiphi)
    phi_L2 = pde.l2.assemble(MESH, space = dxpoly, matrix = 'M', order = order_dphidphi)
    
    
    D_order_dphidphi = pde.int.assemble(MESH, order = order_dphidphi)
    D_order_phiphi = pde.int.assemble(MESH, order = order_phiphi)
    D_order_phiphi_b = pde.int.assembleB(MESH, order = order_phiphi)
    
    
    Kxx = dphix_H1 @ D_order_dphidphi @ dphix_H1.T
    Kyy = dphiy_H1 @ D_order_dphidphi @ dphiy_H1.T
    Cx = phi_L2 @ D_order_dphidphi @ dphix_H1.T
    Cy = phi_L2 @ D_order_dphidphi @ dphiy_H1.T
    
    D_stator_outer = pde.int.evaluateB(MESH, order = order_phiphi, edges = ind_stator_outer)
    B_stator_outer = phi_H1b@ D_stator_outer @ phi_H1b.T
    
    penalty = 1e10
    
    J = 0; # J0 = 0
    for i in range(48):
        J += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : j3[i], regions = np.r_[ind_trig_coils[i]]).diagonal()
        # J0+= pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : j3[i], regions = np.r_[ind_trig_coils[i]]).diagonal()
    J = 0*J
    
    M0 = 0; M1 = 0; M00 = 0
    for i in range(16):
        M0 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
        M1 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
        
        # M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    
    aJ = phi_H1@ D_order_phiphi @J
    
    aM = dphix_H1_order_phiphi@ D_order_phiphi @(-M1) +\
         dphiy_H1_order_phiphi@ D_order_phiphi @(+M0)
    
    aMnew = aM
    
    
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1), M00, u_height=0)
    # fig.show()
    
    # print('Assembling + stuff ', time.monotonic()-tm)
    ##########################################################################################
    
    
    
    ##########################################################################################
    # Solving with Newton
    ##########################################################################################
    
    
    maxIter = 100
    epsangle = 1e-5;
    
    angleCondition = np.zeros(5)
    eps_newton = 1e-8
    factor_residual = 1/2
    mu = 0.0001
    
    def gss(u):
        ux = dphix_H1.T@u; uy = dphiy_H1.T@u
        
        fxx_grad_u_Kxx = dphix_H1 @ D_order_dphidphi @ sps.diags(fxx(ux,uy))@ dphix_H1.T
        fyy_grad_u_Kyy = dphiy_H1 @ D_order_dphidphi @ sps.diags(fyy(ux,uy))@ dphiy_H1.T
        fxy_grad_u_Kxy = dphiy_H1 @ D_order_dphidphi @ sps.diags(fxy(ux,uy))@ dphix_H1.T
        fyx_grad_u_Kyx = dphix_H1 @ D_order_dphidphi @ sps.diags(fyx(ux,uy))@ dphiy_H1.T
        return (fxx_grad_u_Kxx + fyy_grad_u_Kyy + fxy_grad_u_Kxy + fyx_grad_u_Kyx) + penalty*B_stator_outer
        
    def gs(u):    
        ux = dphix_H1.T@u; uy = dphiy_H1.T@u
        return dphix_H1 @ D_order_dphidphi @ fx(ux,uy) + dphiy_H1 @ D_order_dphidphi @ fy(ux,uy) + penalty*B_stator_outer@u - aJ + aM
    
    def J(u):
        ux = dphix_H1.T@u; uy = dphiy_H1.T@u
        return np.ones(D_order_dphidphi.size)@ D_order_dphidphi @f(ux,uy) -(aJ-aM)@u + 1/2*penalty*u@B_stator_outer@u
    
    tm = time.monotonic()
    for i in range(maxIter):
        gsu = gs(u)
        gssu = gss(u)
        # w = chol(gssu).solve_A(-gsu)
        w = sps.linalg.spsolve(gssu,-gsu)
        
        norm_w = np.linalg.norm(w)
        norm_gsu = np.linalg.norm(gsu)
        
        if (-(w@gsu)/(norm_w*norm_gsu)<epsangle):
            angleCondition[i%5] = 1
            if np.product(angleCondition)>0:
                w = -gsu
                print("STEP IN NEGATIVE GRADIENT DIRECTION")
        else: angleCondition[i%5]=0
        
        alpha = 1
        
        # ResidualLineSearch
        # for k in range(1000):
        #     if np.linalg.norm(gs(u+alpha*w)) <= np.linalg.norm(gs(u)): break
        #     else: alpha = alpha*factor_residual
        
        # AmijoBacktracking
        float_eps = np.finfo(float).eps
        for kk in range(1000):
            if J(u+alpha*w)-J(u) <= alpha*mu*(gsu@w) + np.abs(J(u))*float_eps: break
            else: alpha = alpha*factor_residual
            
        u = u + alpha*w
        
        # print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(u)+"|| ||grad||: %2e" %np.linalg.norm(gs(u))+"||alpha: %2e" % (alpha))
        
        if(np.linalg.norm(gs(u)) < eps_newton): break
    
    elapsed = time.monotonic()-tm
    # print('Solving took ', elapsed, 'seconds')
    
    ##########################################################################################
    # Torque computation
    ##########################################################################################
    
    lz = 0.1795 # where does this come from?
    rTorqueOuter = 79.242*10**(-3)
    rTorqueInner = 78.63225*10**(-3)
    
    ind_air_gaps = MESH.getIndices2d(regions_2d, 'air_gap', exact = 0, return_index = True)[0]
    
    Q0 =  pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : x*y/np.sqrt(x**2+y**2), regions = ind_air_gaps).diagonal()
    Q1 =  pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : (y**2-x**2)/(2*np.sqrt(x**2+y**2)), regions = ind_air_gaps).diagonal()
    Q2 =  pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : (y**2-x**2)/(2*np.sqrt(x**2+y**2)), regions = ind_air_gaps).diagonal()
    Q3 =  pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : -x*y/np.sqrt(x**2+y**2), regions = ind_air_gaps).diagonal()
    ux = dphix_H1.T@u; uy = dphiy_H1.T@u
    
    T = lz*nu0/(rTorqueOuter-rTorqueInner) * ((Q0*ux)@D_order_dphidphi@ux +
                                              (Q1*uy)@D_order_dphidphi@ux +
                                              (Q2*ux)@D_order_dphidphi@uy +
                                              (Q3*uy)@D_order_dphidphi@uy)
    print(k,'Torque:', T)
    
    tor[k] = T
    
    
    
    
    plt.cla()
    xs = []
    ys = []
    for ss, tt, uu, vv in zip(MESH.p[MESH.e[:,0],0],
                          MESH.p[MESH.e[:,0],1],
                          MESH.p[MESH.e[:,1],0],
                          MESH.p[MESH.e[:,1],1]):
        xs.append(ss)
        xs.append(uu)
        xs.append(None)
        ys.append(tt)
        ys.append(vv)
        ys.append(None)
        
    ax.plot(xs,ys, linewidth = 1, color = 'red')
    ax.tripcolor(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3], u, cmap = cmap, shading = 'gouraud', edgecolor = 'k', lw = 0.1)
    ax.tricontour(MESH.p[:,0], MESH.p[:,1], u, levels = 25, colors='k', linewidths = 0.5, linestyles = 'solid')
    # ax1.contour(MESH.p[:,0], MESH.p[:,1], u, levels=14, linewidths=0.5, colors='k')
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # time.sleep(0.1)
    # plt.pause(0.01)
    # plt.draw()
    # with writer.saving(fig, "writer_test.mp4", 100):
    writer.grab_frame()
    # print('dsa')
    
    # if k%10 == 50:    
    #     fig = MESH.pdesurf_hybrid(dict(trig = 'P1', quad = 'Q0', controls = 1), u, u_height = 0)
    #     fig.data[0].colorscale='Jet'
    #     fig.data[0].cmax = +0.016
    #     fig.data[0].cmin = -0.016
    #     # fig.show()
    
    ##########################################################################################
    
    # O(n^3/2) complexity for sparse cholesky, done #newton times
    # n = fss(u).shape[0]
    # flops = (n**(3/2)*i)/elapsed
    # print('Approx GFLOP/S',flops*10**(-9))
    
    
    # if k > 6:
        
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1', quad = 'Q1', controls = 0), u[:MESH.np], u_height = 0)
    # # fig.layout.scene.camera.projection.type = "orthographic"
    # fig.data[0].colorscale = 'Jet'
    # fig.show()
    
    # if dxpoly == 'P1':
    #     ux = dphix_H1_o1.T@u
    #     uy = dphiy_H1_o1.T@u
    #     norm_ux = ux**2+uy**2
    #     fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', quad = 'Q1d', controls = 1), norm_ux, u_height = 0)
    #     fig.data[0].colorscale='Jet'
    #     fig.data[0].cmax = 2.5
    #     fig.data[0].cmin = 0
        
    # if dxpoly == 'P0':
    #     ux = dphix_H1_o0.T@u
    #     uy = dphiy_H1_o0.T@u
    #     norm_ux = ux**2+uy**2
    #     fig = MESH.pdesurf_hybrid(dict(trig = 'P0', quad = 'Q0', controls = 1), norm_ux, u_height = 0)
    #     fig.data[0].colorscale='Jet'
    #     fig.data[0].cmax = 2.5
    #     fig.data[0].cmin = 0
        
    # # fig.show()

    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1', quad = 'Q1', controls = 0), u[:MESH.np], u_height = 0)
    # fig.data[0].colorscale='Jet'
    # fig.layout.scene.camera.projection.type = "orthographic"
    # fig.show()
    
    ##########################################################################################
    
    rt += rot_speed
    
    trig_air_gap_rotor = t[np.where(mask_air_gap_rotor)[0],0:3]
    shifted_coeff = np.roll(edges_rotor_outer[:,0],rt)
    kk, jj = MESH._mesh__ismember(trig_air_gap_rotor,edges_rotor_outer[:,0])
    trig_air_gap_rotor[kk] = shifted_coeff[jj]

    p_new = p.copy(); t_new = t.copy()
    p_new[points_rotor,:] = (R(a1*rt)@p[points_rotor,:].T).T
    t_new[np.where(mask_air_gap_rotor)[0],0:3] = trig_air_gap_rotor
    m_new = R(a1*rt)@m
    
    MESH = pde.mesh(p_new,e,t_new,q)

    # fig = MESH.pdemesh()
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 0), f(1,1)+0*vek, u_height=0)
    # fig.show()
    
    # u = np.r_[u[:MESH.np],1/2*(u[MESH.EdgesToVertices[:,0]] + u[MESH.EdgesToVertices[:,1]])].copy()

    
    # MESH.pdemesh2(ax = ax)
    # fig.canvas.draw()
    # fig.canvas.flush_events()
 
writer.finish()