import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../mixed.EM') # adds parent directory
# sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
# from sksparse.cholmod import cholesky as chol
import plotly.io as pio
pio.renderers.default = 'browser'
import numba as nb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
# from matplotlib.animation import FFMpegWriter
# cmap = matplotlib.colors.ListedColormap("limegreen")
cmap = plt.cm.jet


# metadata = dict(title = 'Motor')
# writer = FFMpegWriter(fps = 10, metadata = metadata)

# plt.ion()
plt.close('all')
fig = plt.figure()
fig.show()
ax1 = fig.add_subplot(111)
ax1.set_aspect(aspect = 'equal')
# ax2.set_aspect(aspect = 'equal')

# cbar = plt.colorbar(ax)

# writer.setup(fig, "writer_test.mp4", 500)

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
t = MESH.t
p = MESH.p
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
# Order configuration
##########################################################################################
if ORDER == 1:
    poly = 'P1'
    dxpoly = 'P0'
    order_phiphi = 2
    order_dphidphi = 0
    u = np.random.rand(MESH.np) * 0.5
    
if ORDER == 2:
    poly = 'P2'
    dxpoly = 'P1'
    order_phiphi = 4
    order_dphidphi = 2
    u = np.zeros(MESH.np + MESH.NoEdges)
##########################################################################################



# ##########################################################################################
# # Brauer/Nonlinear laws ... ?
# ##########################################################################################

# k1 = 49.4; k2 = 1.46; k3 = 520.6
# # k1 = 3.8; k2 = 2.17; k3 = 396.2


# f_nonlinear = lambda x,y : k1/(2*k2)*(np.exp(k2*(x**2+y**2))-1) + 1/2*k3*(x**2+y**2) # magnetic energy density in iron

# nu = lambda x,y : k1*np.exp(k2*(x**2+y**2))+k3
# nux = lambda x,y : 2*x*k1*k2*np.exp(k2*(x**2+y**2))
# nuy = lambda x,y : 2*y*k1*k2*np.exp(k2*(x**2+y**2))


# fx_nonlinear = lambda x,y : nu(x,y)*x
# fy_nonlinear = lambda x,y : nu(x,y)*y
# fxx_nonlinear = lambda x,y : nu(x,y) + x*nux(x,y)
# fxy_nonlinear = lambda x,y : x*nuy(x,y)
# fyx_nonlinear = lambda x,y : y*nux(x,y)
# fyy_nonlinear = lambda x,y : nu(x,y) + y*nuy(x,y)


#############################################################################################
# f_linear = lambda x,y : 1/2*nu0*(x**2+y**2)
# fx_linear = lambda x,y : nu0*x
# fy_linear = lambda x,y : nu0*y
# fxx_linear = lambda x,y : nu0 + 0*x
# fxy_linear = lambda x,y : x*0
# fyx_linear = lambda x,y : y*0
# fyy_linear = lambda x,y : nu0 + 0*y
# ###########################################################################################


from nonlinLaws import *
from MaterialLawsBiro import biroTest

f_Biro, df_Biro, ddf_Biro = biroTest(fac = 3, n = 1.3)

def f_nonlinear(ux,uy):
    ret = np.zeros(ux.size)
    for i in range(ux.size):
        ret[i] = f_Biro(ux[i],uy[i])
    return ret
        
def fx_nonlinear(ux,uy):
    ret = np.zeros(ux.size)
    for i in range(ux.size):
        ret[i] = df_Biro(ux[i],uy[i])[0]
    return ret
        
def fy_nonlinear(ux,uy):
    ret = np.zeros(ux.size)
    for i in range(ux.size):
        ret[i] = df_Biro(ux[i],uy[i])[1]
    return ret
        
def fxx_nonlinear(ux,uy):
    ret = np.zeros(ux.size)
    for i in range(ux.size):
        ret[i] = ddf_Biro(ux[i],uy[i])[0,0]
    return ret

def fxy_nonlinear(ux,uy):
    ret = np.zeros(ux.size)
    for i in range(ux.size):
        ret[i] = ddf_Biro(ux[i],uy[i])[0,1]
    return ret

def fyx_nonlinear(ux,uy):
    ret = np.zeros(ux.size)
    for i in range(ux.size):
        ret[i] = ddf_Biro(ux[i],uy[i])[1,0]
    return ret

def fyy_nonlinear(ux,uy):
    ret = np.zeros(ux.size)
    for i in range(ux.size):
        ret[i] = ddf_Biro(ux[i],uy[i])[1,1]
    return ret
###########################################################################################



rot_speed = 1; rt = 0
rots = 1
tor = np.zeros(rots)
tor2 = np.zeros(rots)
tor3 = np.zeros(rots)
tor_vw = np.zeros(rots)
energy = np.zeros(rots)

for k in range(rots):
    
    ##########################################################################################
    # Assembling stuff
    ##########################################################################################
    
    # u = np.zeros(MESH.np)
    
    tm = time.monotonic()
    
    phi_H1  = pde.h1.assemble(MESH, space = poly, matrix = 'M', order = order_phiphi)
    phi_H1_o0  = pde.h1.assemble(MESH, space = poly, matrix = 'M', order = 0)
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

    fem_linear = pde.int.evaluate(MESH, order = 0, regions = ind_linear).diagonal()
    fem_nonlinear = pde.int.evaluate(MESH, order = 0, regions = ind_nonlinear).diagonal()
    fem_rotor = pde.int.evaluate(MESH, order = 0, regions = ind_rotor).diagonal()
    fem_air_gap_rotor = pde.int.evaluate(MESH, order = 0, regions = ind_air_gap_rotor).diagonal()
    
    fem_ind_rotor_outer = pde.int.evaluateB(MESH, order = 0, edges = ind_rotor_outer).diagonal()
    
    penalty = 1e10
    
    Ja = 0; J0 = 0
    for i in range(48):
        Ja += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : j3[i], regions = np.r_[ind_trig_coils[i]]).diagonal()
        J0 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : j3[i], regions = np.r_[ind_trig_coils[i]]).diagonal()
    Ja = 0*Ja
    J0 = 0*J0
    
    M0 = 0; M1 = 0; M00 = 0; M10 = 0
    for i in range(16):
        M0 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
        M1 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
        
        M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
        M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    
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
        float_eps = 1e-11; #float_eps = np.finfo(float).eps
        for kk in range(1000):
            if J(u+alpha*w)-J(u) <= alpha*mu*(gsu@w) + np.abs(J(u))*float_eps: break
            else: alpha = alpha*factor_residual
            
        u = u + alpha*w
        
        print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(u)+"|| ||grad||: %2e" %np.linalg.norm(gs(u))+"||alpha: %2e" % (alpha))
        
        if(np.linalg.norm(gs(u)) < eps_newton): break
    
    elapsed = time.monotonic()-tm
    print('Solving took ', elapsed, 'seconds')
    
    
    ax1.cla()
    ux = dphix_H1.T@u; uy = dphiy_H1.T@u
    MESH.pdesurf2(u, ax = ax1)
    # MESH.pdesurf2(ux**2+uy**2, ax = ax1)
    # MESH.pdegeom(ax = ax1)
    # MESH.pdemesh2(ax = ax1)
    
    fig.show()
    
    plt.pause(0.01)
    # writer.grab_frame()
    
    
    ##########################################################################################
    
    trig_rotor = MESH.t[np.where(fem_rotor)[0],0:3]
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
    
    # trig_rotor = MESH.t[np.where(fem_rotor)[0],0:3]
    # points_rotor = np.unique(trig_rotor)
    
    # rt += rot_speed
    # # rt = 1
    
    # MESH.makeBEO()
    
    # beo = MESH.Boundary_EdgeOrientation
    
    # e0 = MESH.e[:,0]; e1 = MESH.e[:,1]
    # ue0 = 1/2*(e0-e1)*beo + 1/2*(e0+e1)
    # ue1 = 1/2*(e1-e0)*beo + 1/2*(e0+e1)
    # ue = np.c_[ue0,ue1].astype(int)
    
    # fem_ind_rotor_outer = pde.int.evaluateB(MESH, order = 0, edges = ind_rotor_outer).diagonal()
    # # edges_rotor_outer = MESH.e[np.where(fem_ind_rotor_outer)[0],:]
    # edges_rotor_outer  = ue[np.where(fem_ind_rotor_outer)[0],:]
    # edges_rotor_outer2 = edges_rotor_outer.copy()
    
    # for i in range(edges_rotor_outer.shape[0]):
    #     ss = edges_rotor_outer.shape[0]//2
    #     if i%2==0:
    #         edges_rotor_outer2[i,:] = edges_rotor_outer[i//2,:]
    #     else:
    #         edges_rotor_outer2[i,:] = edges_rotor_outer[i//2+ss,:]
    #     print(i//2,i//2+ss)
    
    
    # # edges_rotor_outer = MESH.e[np.where(fem_ind_rotor_outer)[0],:]
    
    # fem_air_gap_rotor = pde.int.evaluate(MESH, order = 0, regions = ind_air_gap_rotor).diagonal()
    # trig_air_gap_rotor = MESH.t[np.where(fem_air_gap_rotor)[0],0:3]
    
    
    # shifted_coeff = np.roll(edges_rotor_outer2[:,0],rt)
    # kk, jj = MESH._mesh__ismember(trig_air_gap_rotor,edges_rotor_outer[:,0])
    # trig_air_gap_rotor[kk] = shifted_coeff[jj]
    
    
    
    
    
    # p_new = p.copy(); t_new = t.copy()
    
    # # rotate all the points in the rotor
    # p_new[points_rotor,:] = (R(a1*rt)@p[points_rotor,:].T).T
    # t_new[np.where(fem_air_gap_rotor)[0],0:3] = trig_air_gap_rotor
    # m_new = R(a1*rt)@m
    
    # MESH = pde.mesh(p_new,e,t_new,q)
    
    # fig = plt.figure(); fig.show(); ax1 = fig.add_subplot(111); MESH.pdegeom(ax=ax1); MESH.pdemesh2(ax=ax1); ax1.set_aspect(aspect = 'equal'); MESH.pdesurf2(fem_air_gap_rotor,ax=ax1)
    
    # fig = MESH.pdemesh()
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 0), f(1,1)+0*vek, u_height=0)
    # fig.show()
    
    # u = np.r_[u[:MESH.np],1/2*(u[MESH.EdgesToVertices[:,0]] + u[MESH.EdgesToVertices[:,1]])].copy()
    
    
    # MESH.pdemesh2(ax = ax)
    # fig.canvas.draw()
    # fig.canvas.flush_events()
 
# writer.finish()
# do()