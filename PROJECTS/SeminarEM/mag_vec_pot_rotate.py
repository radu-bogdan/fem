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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
# cmap = matplotlib.colors.ListedColormap("limegreen")
cmap = plt.cm.jet


metadata = dict(title = 'Motor')
writer = FFMpegWriter(fps = 10, metadata = metadata)

# plt.ion()
plt.close('all')    
fig = plt.figure()
fig.show()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_aspect(aspect = 'equal')
# ax2.set_aspect(aspect = 'equal')

# cbar = plt.colorbar(ax)

# writer.setup(fig, "writer_test.mp4", 500)

# @profile
# def do():
    
##########################################################################################
# Loading mesh
##########################################################################################
motor_npz = np.load('../meshes/motor_fo.npz', allow_pickle = True)

p = motor_npz['p'].T
e = motor_npz['e'].T
t = motor_npz['t'].T
q = np.empty(0)
regions_2d = motor_npz['regions_2d']
regions_1d = motor_npz['regions_1d']
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

# Making sure points are equidistant along the connecting edge.
# a1 = 0.006599984566365112
a1 = 0.00320570678937734
r1 = 0.07863225
ind_edges_rotor_outer = np.where(np.isin(e[:,2],593))[0]
edges_rotor_outer = e[ind_edges_rotor_outer,0:2]
for k in range(edges_rotor_outer.shape[0]):
    p[edges_rotor_outer[k,0],0] = r1*np.cos(a1*(k))
    p[edges_rotor_outer[k,0],1] = r1*np.sin(a1*(k))
##########################################################################################




##########################################################################################
# Parameters
##########################################################################################

ORDER = 1
total = 1

nu0 = 10**7/(4*np.pi)

MESH = pde.mesh(p,e,t,q)
MESH.refinemesh()
MESH.refinemesh()
MESH.refinemesh()

t = MESH.t
p = MESH.p
##########################################################################################



##########################################################################################
# Extract indices
##########################################################################################

ind_stator_outer = MESH.getIndices2d(regions_1d, 'stator_outer', return_index = True)[0]
ind_rotor_outer = MESH.getIndices2d(regions_1d, 'rotor_outer', return_index = True)[0]

ind_edges_rotor_outer = np.where(np.isin(MESH.Boundary_Region,ind_rotor_outer))[0]
edges_rotor_outer = MESH.e[ind_edges_rotor_outer,0:2]

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
# for k in range(edges_rotor_outer.shape[0]):
#     MESH.p[edges_rotor_outer[k,0],0] = r1*np.cos(a1*(k))
#     MESH.p[edges_rotor_outer[k,0],1] = r1*np.sin(a1*(k))
##########################################################################################



##########################################################################################
# Order configuration
##########################################################################################
if ORDER == 1:
    poly = 'P1'
    dxpoly = 'P0'
    order_phi = 1
    order_dphi = 0
    order_phiphi = order_phi*2
    order_dphidphi = order_dphi*2
    u = np.zeros(MESH.np)
    
if ORDER == 2:
    poly = 'P2'
    dxpoly = 'P1'
    order_phi = 2
    order_dphi = 1
    order_phiphi = order_phi*2
    order_dphidphi = order_dphi*2
    u = np.zeros(MESH.np + MESH.NoEdges)
##########################################################################################



##########################################################################################
# Brauer/Nonlinear laws ... ?
##########################################################################################

sys.path.insert(1,'../mixed.EM')
from nonlinLaws import *


###########################################################################################

rot_speed = 1; rt = 0
rots = 100
tor = np.zeros(rots)
tor2 = np.zeros(rots)
tor3 = np.zeros(rots)
tor_vw = np.zeros(rots)
energy = np.zeros(rots)

for k in range(rots):
    
    ##########################################################################################
    # Assembling stuff
    ##########################################################################################
    
    # u = np.zeros(MESH.np + MESH.NoEdges)
    
    # u = u/10000
    
    tm = time.monotonic()
    
    phi_H1  = pde.h1.assemble(MESH, space = poly, matrix = 'M', order = order_phiphi)
    phi_H1_order_dphidphi  = pde.h1.assemble(MESH, space = poly, matrix = 'M', order = order_dphidphi)
    phi_H1_order_dphi  = pde.h1.assemble(MESH, space = poly, matrix = 'M', order = order_dphi)
    phi_H1_o0  = pde.h1.assemble(MESH, space = poly, matrix = 'M', order = 0)
    dphix_H1, dphiy_H1 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_dphidphi)
    dphix_H1_order_dphi, dphiy_H1_order_dphi = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_dphi)
    dphix_H1_o0, dphiy_H1_o0 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = 0)
    dphix_H1_o1, dphiy_H1_o1 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = 1)
    dphix_H1_order_phiphi, dphiy_H1_order_phiphi = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_phiphi)
    phi_H1b = pde.h1.assembleB(MESH, space = poly, matrix = 'M', shape = phi_H1.shape, order = order_phiphi)
    phi_L2 = pde.l2.assemble(MESH, space = dxpoly, matrix = 'M', order = order_dphidphi)
    
    
    D_order_dphidphi = pde.int.assemble(MESH, order = order_dphidphi)
    D_order_dphi = pde.int.assemble(MESH, order = order_dphi)
    D_order_phiphi = pde.int.assemble(MESH, order = order_phiphi)
    D_order_phiphi_b = pde.int.assembleB(MESH, order = order_phiphi)
    
    
    Kxx = dphix_H1 @ D_order_dphidphi @ dphix_H1.T
    Kyy = dphiy_H1 @ D_order_dphidphi @ dphiy_H1.T
    Cx = phi_L2 @ D_order_dphidphi @ dphix_H1.T
    Cy = phi_L2 @ D_order_dphidphi @ dphiy_H1.T
    
    D_stator_outer = pde.int.evaluateB(MESH, order = order_phiphi, edges = ind_stator_outer)
    B_stator_outer = phi_H1b@ D_stator_outer @ phi_H1b.T

    fem_linear = pde.int.evaluate(MESH, order = order_dphidphi, regions = ind_linear).diagonal()
    fem_nonlinear = pde.int.evaluate(MESH, order = order_dphidphi, regions = ind_nonlinear).diagonal()
    fem_rotor = pde.int.evaluate(MESH, order = order_dphidphi, regions = ind_rotor).diagonal()
    fem_air_gap_rotor = pde.int.evaluate(MESH, order = order_dphidphi, regions = ind_air_gap_rotor).diagonal()
    
    penalty = 1e10
    
    Ja = 0; Ja0 = 0
    for i in range(48):
        Ja += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : j3[i], regions = np.r_[ind_trig_coils[i]]).diagonal()
        Ja0 += pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : j3[i], regions = np.r_[ind_trig_coils[i]]).diagonal()
    Ja = 0*Ja
    Ja0 = 0*Ja0
    
    M0 = 0; M1 = 0; M00 = 0; M10 = 0; M0_dphi = 0; M1_dphi = 0
    for i in range(16):
        M0 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
        M1 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
        
        M00 += pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
        M10 += pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
        
        M0_dphi += pde.int.evaluate(MESH, order = order_dphi, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
        M1_dphi += pde.int.evaluate(MESH, order = order_dphi, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    
    aJ = phi_H1@ D_order_phiphi @Ja
    
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
        
        fxx_grad_u_Kxx = dphix_H1 @ D_order_dphidphi @ sps.diags(fxx_linear(ux,uy)*fem_linear + fxx_nonlinear(ux,uy)*fem_nonlinear)@ dphix_H1.T
        fyy_grad_u_Kyy = dphiy_H1 @ D_order_dphidphi @ sps.diags(fyy_linear(ux,uy)*fem_linear + fyy_nonlinear(ux,uy)*fem_nonlinear)@ dphiy_H1.T
        fxy_grad_u_Kxy = dphiy_H1 @ D_order_dphidphi @ sps.diags(fxy_linear(ux,uy)*fem_linear + fxy_nonlinear(ux,uy)*fem_nonlinear)@ dphix_H1.T
        fyx_grad_u_Kyx = dphix_H1 @ D_order_dphidphi @ sps.diags(fyx_linear(ux,uy)*fem_linear + fyx_nonlinear(ux,uy)*fem_nonlinear)@ dphiy_H1.T
        return (fxx_grad_u_Kxx + fyy_grad_u_Kyy + fxy_grad_u_Kxy + fyx_grad_u_Kyx) + penalty*B_stator_outer
        
    def gs(u):    
        ux = dphix_H1.T@u; uy = dphiy_H1.T@u
        return dphix_H1 @ D_order_dphidphi @ (fx_linear(ux,uy)*fem_linear + fx_nonlinear(ux,uy)*fem_nonlinear) +\
               dphiy_H1 @ D_order_dphidphi @ (fy_linear(ux,uy)*fem_linear + fy_nonlinear(ux,uy)*fem_nonlinear) + penalty*B_stator_outer@u - aJ + aM
    
    def J(u):
        ux = dphix_H1.T@u; uy = dphiy_H1.T@u
        return np.ones(D_order_dphidphi.size)@ D_order_dphidphi @(f_linear(ux,uy)*fem_linear + f_nonlinear(ux,uy)*fem_nonlinear) -(aJ-aM)@u + 1/2*penalty*u@B_stator_outer@u
    
    tm = time.monotonic()
    for i in range(maxIter):
        gsu = gs(u)
        gssu = gss(u)
        w = chol(gssu).solve_A(-gsu)
        # w = sps.linalg.spsolve(gssu,-gsu)
        
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
        
        print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(u)+"|| ||grad||: %2e" %np.linalg.norm(gs(u))+"||alpha: %2e" % (alpha))
        
        if(np.linalg.norm(gs(u)) < eps_newton): break
    
    elapsed = time.monotonic()-tm
    print('Solving took ', elapsed, 'seconds')
    
    
    
    ##########################################################################################
    # Arkkio torque computation
    ##########################################################################################
    
    # lz = 0.1795 # where does this come from?
    lz = 1
    rTorqueOuter = 79.242*10**(-3)
    rTorqueInner = 78.63225*10**(-3)
    
    ind_air_gaps = MESH.getIndices2d(regions_2d, 'air_gap', exact = 0, return_index = True)[0]
    
    Q0 = pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : x*y/np.sqrt(x**2+y**2), regions = ind_air_gaps).diagonal()
    Q1 = pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : (y**2-x**2)/(2*np.sqrt(x**2+y**2)), regions = ind_air_gaps).diagonal()
    Q2 = pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : (y**2-x**2)/(2*np.sqrt(x**2+y**2)), regions = ind_air_gaps).diagonal()
    Q3 = pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : -x*y/np.sqrt(x**2+y**2), regions = ind_air_gaps).diagonal()
    ux = dphix_H1.T@u; uy = dphiy_H1.T@u
    
    T = lz*nu0/(rTorqueOuter-rTorqueInner) * ((Q0*ux)@D_order_dphidphi@ux +
                                              (Q1*uy)@D_order_dphidphi@ux +
                                              (Q2*ux)@D_order_dphidphi@uy +
                                              (Q3*uy)@D_order_dphidphi@uy)
    print(k, 'Torque:', T)
    
    tor[k] = T
    
    ##########################################################################################
    # Magnetic energy
    ##########################################################################################
    
    energy[k] = J(u)
    
    ##########################################################################################
    # Local energy
    ##########################################################################################
    
    # ux = dphix_H1_order_dphi.T@u
    # uy = dphiy_H1_order_dphi.T@u
    # u_Pk = phi_H1_order_dphi.T@u
    
    # eu = f(ux,uy)-Ja0*u_Pk +(M10*uy - M10*ux)
    # nH = np.sqrt((fx(ux,uy)-M00)**2+(fy(ux,uy)-M10)**2)
    
    
    ##########################################################################################
    # Virtual work principle v2.0
    ##########################################################################################
    
    
    
    fem_linear = pde.int.evaluate(MESH, order = order_dphi, regions = ind_linear).diagonal()
    fem_nonlinear = pde.int.evaluate(MESH, order = order_dphi, regions = ind_nonlinear).diagonal()
    
    ux = dphix_H1_order_dphi.T@u; uy = dphiy_H1_order_dphi.T@u
    f = lambda ux,uy : f_linear(ux,uy)*fem_linear + f_nonlinear(ux,uy)*fem_nonlinear
    fx = lambda ux,uy : fx_linear(ux,uy)*fem_linear + fx_nonlinear(ux,uy)*fem_nonlinear
    fy = lambda ux,uy : fy_linear(ux,uy)*fem_linear + fy_nonlinear(ux,uy)*fem_nonlinear
    
    u_Pk = phi_H1_order_dphi.T@u
    
    # r1 = p[edges_rotor_outer[0,0],0]
    # r2 = r1 + 0.0007
    # r2 = r1 + 0.00024
    
    #outer radius rotor
    r_outer = 78.63225*10**(-3);
    #sliding mesh rotor
    r_sliding = 78.8354999*10**(-3);
    #sliding mesh stator
    r_sliding2 = 79.03874999*10**(-3);
    #inner radius stator
    r_inner = 79.242*10**(-3);
    
    r1 = r_outer
    r2 = r_inner
    
    # dazwischen = lambda x,y : 
    scale = lambda x,y : 1*(x**2+y**2<r1**2)+(x**2+y**2-r2**2)/(r1**2-r2**2)*(x**2+y**2>r1**2)*(x**2+y**2<r2**2)
    scalex = lambda x,y : (2*x)/(r1**2-r2**2)#*(x**2+y**2>r1**2)*(x**2+y**2<r2**2)
    scaley = lambda x,y : (2*y)/(r1**2-r2**2)#*(x**2+y**2>r1**2)*(x**2+y**2<r2**2)
    
    v = lambda x,y : np.r_[-y,x]*scale(x,y) # v wird nie wirklich gebraucht...
    
    v1x = lambda x,y : -y*scalex(x,y)
    v1y = lambda x,y : -scale(x,y)-y*scaley(x,y)
    v2x = lambda x,y :  scale(x,y)+x*scalex(x,y)
    v2y = lambda x,y :  x*scaley(x,y)
    
    
    v1x_fem = pde.int.evaluate(MESH, order = order_dphi, coeff = v1x, regions = np.r_[ind_air_gaps]).diagonal()
    v1y_fem = pde.int.evaluate(MESH, order = order_dphi, coeff = v1y, regions = np.r_[ind_air_gaps]).diagonal()
    v2x_fem = pde.int.evaluate(MESH, order = order_dphi, coeff = v2x, regions = np.r_[ind_air_gaps]).diagonal()
    v2y_fem = pde.int.evaluate(MESH, order = order_dphi, coeff = v2y, regions = np.r_[ind_air_gaps]).diagonal()
    
    scale_fem = pde.int.evaluate(MESH, order = order_dphi, coeff = scale, regions = np.r_[ind_air_gaps,ind_rotor]).diagonal()
    
    one_fem = pde.int.evaluate(MESH, order = order_dphi, coeff = lambda x,y : 1+0*x+0*y, regions = np.r_[ind_air_gaps,ind_rotor]).diagonal()
    # v1x_fem = pde.int.evaluate(MESH, order = order_dphi, coeff = v1x, regions = ind_air_gaps).diagonal()
    
    
    b1 = uy
    b2 = -ux
    fu = f(ux,uy)+1/2*(M0_dphi*uy-M1_dphi*ux)
    fbb1 =  fy(ux,uy)+M0_dphi
    fbb2 = -fx(ux,uy)+M1_dphi
    a_Pk = u_Pk
    
    
    term1 = (fu - fbb1*b1 -fbb2*b2 -Ja0*a_Pk)*(v1x_fem + v2y_fem)
    term2 = (fbb1*b1)*v1x_fem + (fbb2*b1)*v2x_fem + (fbb1*b2)*v1y_fem + (fbb2*b2)*v2y_fem
    
    term_2 = -(term1+term2)
    tor2[k] = one_fem@D_order_dphi@term_2
    print(k, 'Torque2:', tor2[k])
    
    b1 = uy
    b2 = -ux
    fu = f(ux,uy)
    fbb1 =  fy(ux,uy)
    fbb2 = -fx(ux,uy)
    a_Pk = u_Pk
    
    
    term1 = (fu - fbb1*b1 -fbb2*b2 -Ja0*a_Pk)*(v1x_fem + v2y_fem)
    term2 = (fbb1*b1)*v1x_fem + (fbb2*b1)*v2x_fem + (fbb1*b2)*v1y_fem + (fbb2*b2)*v2y_fem
    
    term_3 = -(term1+term2)
    tor3[k] = one_fem@D_order_dphi@term_3
    print(k, 'Torque3:', tor3[k])
    
    
    fig = MESH.pdesurf(v1x_fem)
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1), (ux-1/nu0*M1_dphi)**2+(uy+1/nu0*M0_dphi)**2, u_height = 0)
    fig.show()
    
    # Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
    # stop
    
    
    # stiffness_part2 = gs(u)@u
    
    # print("loale: ",stiffness_part2-stiffness_part)
    
    # chip = ax1.tripcolor(Triang, (ux-1/nu0*M1_dphi)**2+(uy+1/nu0*M0_dphi)**2, cmap = cmap, shading = 'flat', lw = 0.1)
    
    
    
    
    ax1.cla()
    ax2.cla()
    
    # MESH.pdegeom(ax = ax1)
    # MESH.pdemesh2(ax = ax1)
    
    Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
    
    # chip = ax1.tripcolor(Triang, -stiffness_part_local + energy_part_local, cmap = cmap, shading = 'flat', lw = 0.1)
    # ax1.tripcolor(Triang, zT, cmap = cmap, shading = 'flat', lw = 0.1)
    # chip = ax1.tripcolor(Triang, u, cmap = cmap, shading = 'gouraud', lw = 0.1)
    # chip = ax1.tripcolor(Triang, fx(ux,uy)**2+fy(ux,uy)**2, cmap = cmap, shading = 'flat', lw = 0.1)
    # chip = ax1.tripcolor(Triang, v1y_fem, cmap = cmap, shading = 'flat', lw = 0.1)
    # chip = ax1.tripcolor(Triang, ux**2+uy**2, cmap = cmap, shading = 'flat', lw = 0.1)
    # chip = ax1.tripcolor(Triang, term_3, cmap = cmap, shading = 'flat')
    # chip = ax1.tripcolor(Triang, np.abs(term), cmap = cmap, shading = 'flat')
    # chip = ax1.tripcolor(Triang, eu, cmap = cmap, shading = 'flat', lw = 0.1, norm=colors.LogNorm(vmin=np.abs(eu.min()), vmax=np.abs(eu.max())))
    # ax1.tricontour(Triang, u, levels = 25, colors = 'k', linewidths = 0.5, linestyles = 'solid') #Feldlinien
    
    # ax1.plot(tor,linewidth = 1)
    # ax1.plot(tor2,linewidth = 1)
    
    energy_diff = energy[1:]-np.r_[energy[-1],energy[0:-2]]
    energy_diff = (-energy_diff*(np.abs(energy_diff)<1)/(2*a1)).copy()
    
    # ax1.plot(energy_diff,linewidth = 1)
    
    # ax2.plot(energy_diff-tor[:-1])
    # ax2.plot(energy_diff-tor2[:-1])
    # ax2.plot(tor-tor2)
    # ax2.plot(tor-tor3)
    
    ax2.plot(energy_diff)
    ax2.plot(tor)
    ax2.plot(tor2)
    ax2.plot(tor3)
    
    # ax2.plot(tor/tor2)
    
    # fig.show()
    
    plt.pause(0.01)
    # writer.grab_frame()
    
    
    
    
    # ax.tripcolor(Triang, u, cmap = cmap, shading = 'gouraud', edgecolor = 'k', lw = 0.1)
    
    # ax.tripcolor(Triang, np.sqrt(ux**2+uy**2), shading = 'flat', edgecolor = 'k', lw = 0.1)
    
    # xx_trig = np.c_[MESH.p[MESH.t[:,0],0],MESH.p[MESH.t[:,1],0],MESH.p[MESH.t[:,2],0]]
    # yy_trig = np.c_[MESH.p[MESH.t[:,0],1],MESH.p[MESH.t[:,1],1],MESH.p[MESH.t[:,2],1]]
    
    # xxx_trig = np.c_[xx_trig,xx_trig[:,0],np.nan*xx_trig[:,0]]
    # yyy_trig = np.c_[yy_trig,yy_trig[:,0],np.nan*yy_trig[:,0]]
        
    # ax.plot(
    #     xxx_trig.flatten(), 
    #     yyy_trig.flatten(),
    #     color = 'k',
    #     linewidth = 0.1
    # )
    
    # MESH.pdemesh2(ax = ax)
    
    # ax.set_xlim(left = -0.081, right = -0.065)
    # ax.set_ylim(bottom = 0.0015, top = 0.006)
    
    # chip = ax1.tripcolor(Triang, np.sqrt(ux**2+uy**2), cmap = cmap, shading = 'flat', lw = 0.1, vmin = 0, vmax = 2.3)
    
    # nH = np.sqrt((fx(ux,uy)-M00)**2+(fy(ux,uy)-M10)**2)
    
    # ux = dphix_H1_o0.T@u
    # uy = dphix_H1_o0.T@u
    # u_P0 = 1/3*(u[MESH.t[:,0]]+u[MESH.t[:,1]]+u[MESH.t[:,2]])
    # eu = f(ux,uy)-J0*u_P0 - M00*uy + M10*ux
    
    # plt.pause(0.00001)
    
    # if k == 0:
    #     cbar = plt.colorbar(chip)
    # else:
    #     cbar.update_normal(chip)
    
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
    
    stop
    
    fem_rotor = pde.int.evaluate(MESH, order = 0, regions = ind_rotor).diagonal()
    fem_air_gap_rotor = pde.int.evaluate(MESH, order = 0, regions = ind_air_gap_rotor).diagonal()
    
    trig_rotor = t[np.where(fem_rotor)[0],0:3]
    points_rotor = np.unique(trig_rotor)
    
    rt += rot_speed
    
    trig_air_gap_rotor = t[np.where(fem_air_gap_rotor)[0],0:3]
    shifted_coeff = np.roll(edges_rotor_outer[:,0],rt)
    kk, jj = MESH._mesh__ismember(trig_air_gap_rotor,edges_rotor_outer[:,0])
    trig_air_gap_rotor[kk] = shifted_coeff[jj]

    p_new = p.copy(); t_new = t.copy()
    p_new[points_rotor,:] = (R(a1*rt)@p[points_rotor,:].T).T
    t_new[np.where(fem_air_gap_rotor)[0],0:3] = trig_air_gap_rotor
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
# do()