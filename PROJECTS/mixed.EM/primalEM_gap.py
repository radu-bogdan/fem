import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../mixed.EM') # adds parent directory
# sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
from sksparse.cholmod import cholesky as chol
import plotly.io as pio
pio.renderers.default = 'browser'

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
# cmap = matplotlib.colors.ListedColormap("limegreen")
cmap = plt.cm.jet

metadata = dict(title = 'Motor')
writer = FFMpegWriter(fps = 15, metadata = metadata)

##########################################################################################
# Loading mesh
##########################################################################################

print('loaded stuff...')

motor_npz = np.load('../meshes/motor_pizza_gap.npz', allow_pickle = True)

m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']
##########################################################################################

print('loaded geo...')

##########################################################################################
# Parameters
##########################################################################################

ORDER = 1
total = 1

nu0 = 10**7/(4*np.pi)


import ngsolve as ng
import netgen.occ as occ

geoOCC = motor_npz['geoOCC'].tolist()
geoOCCmesh = geoOCC.GenerateMesh()
ngsolvemesh = ng.Mesh(geoOCCmesh)
# ngsolvemesh.Refine()
# ngsolvemesh.Refine()

MESH = pde.mesh.netgen(ngsolvemesh.ngmesh)

linear = '*air,*magnet,shaft_iron,*coil'
nonlinear = 'stator_iron,rotor_iron'

def getPoints(MESH):
    airL_index = MESH.getIndices2d(MESH.regions_1d,'airL')[0]
    airR_index = MESH.getIndices2d(MESH.regions_1d,'airR')[0]
    
    pointsL_index = np.unique(MESH.e[np.argwhere(MESH.e[:,2] == airL_index)[:,0],:2])
    pointsR_index = np.unique(MESH.e[np.argwhere(MESH.e[:,2] == airR_index)[:,0],:2])
    
    pointsL = MESH.p[pointsL_index,:]; pointsR = MESH.p[pointsR_index,:]
    
    indL = np.argsort(pointsL[:,0]**2) # es reicht, nach der ersten Koordinate zu sortieren!
    indR = np.argsort(pointsR[:,0]**2)
    
    pointsL_index_sorted = pointsL_index[indL]
    pointsR_index_sorted = pointsR_index[indR]
    
    edges0 = np.c_[pointsL_index_sorted[:-1],
                   pointsL_index_sorted[1:]]
                   
    edges1 = np.c_[pointsR_index_sorted[:-1],
                   pointsR_index_sorted[1:]]
    
    edges0 = np.sort(edges0)
    edges1 = np.sort(edges1)

    edgecoord0 = np.zeros(edges0.shape[0],dtype=int)-1
    edgecoord1 = np.zeros(edges1.shape[0],dtype=int)-1
    
    for i in range(edges0.shape[0]):
        v0 =  np.where(np.all(MESH.EdgesToVertices[:,:2]==edges0[i,:],axis=1))[0]
        v1 =  np.where(np.all(MESH.EdgesToVertices[:,:2]==edges1[i,:],axis=1))[0]
        edgecoord0[i] = v0
        edgecoord1[i] = v1
    
    ident_points_gap = np.c_[pointsL_index_sorted,
                             pointsR_index_sorted]
    
    ident_edges_gap = np.c_[edgecoord0,
                            edgecoord1]
    
    return ident_points_gap, ident_edges_gap

ident_points_gap, ident_edges_gap = getPoints(MESH)

def makeIdentifications(MESH):

    a = np.array(MESH.geoOCCmesh.GetIdentifications())

    c0 = np.zeros(a.shape[0])
    c1 = np.zeros(a.shape[0])

    for i in range(a.shape[0]):
        point0 = MESH.p[a[i,0]-1]
        point1 = MESH.p[a[i,1]-1]

        c0[i] = point0[0]**2+point0[1]**2
        c1[i] = point1[0]**2+point1[1]**2

    ind0 = np.argsort(c0)

    aa = np.c_[a[ind0[:-1],0]-1,
               a[ind0[1: ],0]-1]

    edges0 = np.c_[a[ind0[:-1],0]-1,
                   a[ind0[1: ],0]-1]
    edges1 = np.c_[a[ind0[:-1],1]-1,
                   a[ind0[1: ],1]-1]

    edges0 = np.sort(edges0)
    edges1 = np.sort(edges1)

    edgecoord0 = np.zeros(edges0.shape[0],dtype=int)-1
    edgecoord1 = np.zeros(edges1.shape[0],dtype=int)-1
    
    for i in range(edges0.shape[0]):
        v0 =  np.where(np.all(MESH.EdgesToVertices[:,:2]==edges0[i,:],axis=1))[0]
        v1 =  np.where(np.all(MESH.EdgesToVertices[:,:2]==edges1[i,:],axis=1))[0]
        
        if v0.size == 1:
            edgecoord0[i] = v0
            edgecoord1[i] = v1
    
    
    identification = np.c_[np.r_[a[ind0,0]-1,MESH.np + edgecoord0],
                           np.r_[a[ind0,1]-1,MESH.np + edgecoord1]]
    ident_points = np.c_[a[ind0,0]-1,
                         a[ind0,1]-1]
    ident_edges = np.c_[edgecoord0,
                        edgecoord1]
    
    index = np.argwhere((ident_edges[:,0] == -1)*(ident_edges[:,1] == -1))[0]
    if index.size ==1:
        ident_edges = np.delete(ident_edges, index, axis=0)

    return ident_points, ident_edges

ident_points, ident_edges = makeIdentifications(MESH)

print('generated mesh...')

##########################################################################################
# Order configuration
##########################################################################################
if ORDER == 1:
    poly = 'P1'
    dxpoly = 'P0'
    order_phiphi = 2
    order_dphidphi = 0
    # u = np.random.rand(MESH.np) * 0.005
    u = np.zeros(MESH.np)#+0.01
    
if ORDER == 2:
    poly = 'P2'
    dxpoly = 'P1'
    order_phiphi = 4
    order_dphidphi = 2
    u = np.zeros(MESH.np + MESH.NoEdges)
############################################################################################



############################################################################################
## Brauer/Nonlinear laws ... ?
############################################################################################

sys.path.insert(1,'../mixed.EM')
from nonlinLaws import *
                       
############################################################################################

rot_speed = 1;
rots = 305
tor = np.zeros(rots)
tor2 = np.zeros(rots)
tor3 = np.zeros(rots)
tor_vw = np.zeros(rots)
energy = np.zeros(rots)

for k in range(rots):
    
    print('Step : ', k)
    
    ##########################################################################################
    # Assembling stuff
    ##########################################################################################
    
    u = np.zeros(MESH.np)
   
    tm = time.monotonic()
    
    phi_H1  = pde.h1.assemble(MESH, space = poly, matrix = 'M', order = order_phiphi)
    phi_H1_o0  = pde.h1.assemble(MESH, space = poly, matrix = 'M', order = 0)
    dphix_H1, dphiy_H1 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_dphidphi)
    dphix_H1_o0, dphiy_H1_o0 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = 0)
    dphix_H1_o1, dphiy_H1_o1 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = 1)
    dphix_H1_order_phiphi, dphiy_H1_order_phiphi = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_phiphi)
    phi_H1b = pde.h1.assembleB(MESH, space = poly, matrix = 'M', shape = phi_H1.shape, order = order_phiphi)
    phi_L2 = pde.l2.assemble(MESH, space = dxpoly, matrix = 'M', order = order_dphidphi)
    
    R0, RSS = pde.h1.assembleR(MESH, space = poly, edges = 'stator_outer')
    
    ##########################################################################################
    # Identifications
    ##########################################################################################
    
    if ORDER == 1:
        ident = ident_points
    if ORDER == 2:
        ident = np.r_[ident_points, MESH.np + ident_edges]
    
    i0 = ident[:,0]; i1 = ident[:,1]
    
    R_out, R_int = pde.h1.assembleR(MESH, space = poly, edges = 'stator_outer,left,right,airL,airR')
    R_L, R_LR = pde.h1.assembleR(MESH, space = poly, edges = 'left', listDOF = i1)
    R_R, R_RR = pde.h1.assembleR(MESH, space = poly, edges = 'right', listDOF = i0)
    
    # R_0, R_0R = pde.h1.assembleR(MESH, space = poly, edges = 'stator_outer')
    
    # ident_gap = np.c_[np.roll(ident_points_gap[:,0],-k*rot_speed),
    #                           ident_points_gap[:,1]]
    # if ORDER == 2:
                                     
    #     # ident_points_gap = np.c_[ident_edges_gap[:,0],
    #     #                          np.roll(ident_edges_gap[:,1],1)]
    #     ident_edges_gap = np.c_[np.roll(ident_edges_gap[:,0],-k*rot_speed),
    #                              ident_edges_gap[:,1]]
    #     ident_gap = np.r_[ident_points_gap, MESH.np + ident_edges_gap]
        
        
        
        
    i0_gap = np.roll(ident_points_gap[:,0], -k*rot_speed)
    i1_gap = ident_points_gap[:,1]
    
    R_AL, R_ALR = pde.h1.assembleR(MESH, space = poly, edges = 'airL', listDOF = i0_gap)
    R_AR, R_ARR = pde.h1.assembleR(MESH, space = poly, edges = 'airR', listDOF = i1_gap)
    
     
    
    # manual stuff: (removing the point in the three corners...)
    corners = np.r_[0,16,17,ident_points.shape[0]-1] #,21,24
    ind1 = np.setdiff1d(np.r_[0:R_L.shape[0]], corners)
    R_L = R_L[ind1,:]
    
    corners = np.r_[0,16,17,ident_points.shape[0]-1] #,22,23
    ind1 = np.setdiff1d(np.r_[0:R_R.shape[0]], corners)
    R_R = R_R[ind1,:]
    
    # corners_gap = np.r_[0,ident_points_gap.shape[0]-1]
    # corners_gap = np.r_[0,ident_points_gap.shape[0]-1]
    # ind1 = np.setdiff1d(np.r_[0:R_AL.shape[0]], corners_gap)
    # R_AL = R_AL[100:200,:]
    # R_AR = R_AR[100:200,:]
    
    if k>0:
        R_AL[-k*rot_speed:,:] = -R_AL[-k*rot_speed:,:]
    
    from scipy.sparse import bmat
    RS =  bmat([[R_int], [R_L-R_R], [R_AL+R_AR]])
    ##########################################################################################
    
    
        
    D_order_dphidphi = pde.int.assemble(MESH, order = order_dphidphi)
    D_order_phiphi = pde.int.assemble(MESH, order = order_phiphi)
    D_order_phiphi_b = pde.int.assembleB(MESH, order = order_phiphi)
    
    
    Kxx = dphix_H1 @ D_order_dphidphi @ dphix_H1.T
    Kyy = dphiy_H1 @ D_order_dphidphi @ dphiy_H1.T
    Cx = phi_L2 @ D_order_dphidphi @ dphix_H1.T
    Cy = phi_L2 @ D_order_dphidphi @ dphiy_H1.T
    
    D_stator_outer = pde.int.evaluateB(MESH, order = order_phiphi, edges = 'stator_outer')
    B_stator_outer = phi_H1b@ D_stator_outer @ phi_H1b.T

    fem_linear = pde.int.evaluate(MESH, order = order_dphidphi, regions = linear).diagonal()
    fem_nonlinear = pde.int.evaluate(MESH, order = order_dphidphi, regions = nonlinear).diagonal()
    
    penalty = 1e10
    
    Ja = 0; J0 = 0
    for i in range(48):
        Ja += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : j3[i], regions = 'coil'+str(i+1)).diagonal()
        J0 += pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : j3[i], regions = 'coil'+str(i+1)).diagonal()
    Ja = 0*Ja
    J0 = 0*J0
    
    M0 = 0; M1 = 0; M00 = 0; M10 = 0; M1_dphi = 0; M0_dphi = 0
    for i in range(16):
        M0 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
        M1 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
        
        M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
        M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
        
        M0_dphi += pde.int.evaluate(MESH, order = 1, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
        M1_dphi += pde.int.evaluate(MESH, order = 1, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
    
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
        return (fxx_grad_u_Kxx + fyy_grad_u_Kyy + fxy_grad_u_Kxy + fyx_grad_u_Kyx) #+ penalty*B_stator_outer
        
    def gs(u):
        ux = dphix_H1.T@u; uy = dphiy_H1.T@u
        return dphix_H1 @ D_order_dphidphi @ (fx_linear(ux,uy)*fem_linear + fx_nonlinear(ux,uy)*fem_nonlinear) +\
               dphiy_H1 @ D_order_dphidphi @ (fy_linear(ux,uy)*fem_linear + fy_nonlinear(ux,uy)*fem_nonlinear) - aJ + aM #+ penalty*B_stator_outer@u 
    
    def J(u):
        ux = dphix_H1.T@u; uy = dphiy_H1.T@u
        return np.ones(D_order_dphidphi.size)@ D_order_dphidphi @(f_linear(ux,uy)*fem_linear + f_nonlinear(ux,uy)*fem_nonlinear) -(aJ-aM)@u #+ 1/2*penalty*u@B_stator_outer@u
    
    tm2 = time.monotonic()
    for i in range(maxIter):
        # gssu = gss(u)
        # gsu = gs(u)
        
        gssu = RS @ gss(u) @ RS.T
        gsu = RS @ gs(u)
        
        tm = time.monotonic()
        wS = chol(gssu).solve_A(-gsu)
        # wS = sps.linalg.spsolve(gssu,-gsu)
        print('Solving took ', time.monotonic()-tm)
        
        # w = wS
        w = RS.T@wS
        
        # norm_w = np.linalg.norm(w,np.inf)
        # norm_gsu = np.linalg.norm(gsu,np.inf)
        # if (-(wS@gsu)/(norm_w*norm_gsu)<epsangle):
        #     angleCondition[i%5] = 1
        #     if np.product(angleCondition)>0:
        #         w = -gsu
        #         print("STEP IN NEGATIVE GRADIENT DIRECTION")
        # else: angleCondition[i%5]=0
        
        alpha = 1
        
        # ResidualLineSearch
        # for k in range(1000):
        #     if np.linalg.norm(gs(u+alpha*w),np.inf) <= np.linalg.norm(gs(u),np.inf): break
        #     else: alpha = alpha*factor_residual
        
        # AmijoBacktracking
        float_eps = 1e-11; #float_eps = np.finfo(float).eps
        for kk in range(1000):
            if J(u+alpha*w)-J(u) <= alpha*mu*(gsu@wS) + np.abs(J(u))*float_eps: break
            else: alpha = alpha*factor_residual
            
        u = u + alpha*w
        
        print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(u)+"|| ||grad||: %2e" %np.linalg.norm(RS @ gs(u),np.inf)+"||alpha: %2e" % (alpha))
        
        if(np.linalg.norm(RS @ gs(u)) < eps_newton): break
    
    elapsed = time.monotonic()-tm2
    print('Solving took ', elapsed, 'seconds')
    
    
    # ax1.cla()
    ux = dphix_H1_o1.T@u; uy = dphiy_H1_o1.T@u
    
    # fig = MESH.pdesurf((ux-1/nu0*M1_dphi)**2+(uy+1/nu0*M0_dphi)**2, u_height = 0, cmax = 5)
    # fig.show()
    
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1',quad = 'Q0',controls = 1), u[0:MESH.np], u_height=1)
    # fig.show()
    
    
    # if k == 0:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    
    # ax.cla()
    # MESH.pdesurf2(u[:MESH.np],ax = ax)
    
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # time.sleep(0.1)
    
    # input()
    
    if k == 0:
        fig = plt.figure()
        writer.setup(fig, "writer_test.mp4", 500)
        fig.show()
        
    ax = fig.add_subplot(111)
    ax.set_aspect(aspect = 'equal')
    MESH.pdesurf2(u,ax = ax)
    # MESH.pdemesh2(ax = ax)
    MESH.pdegeom(ax = ax)
    Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
    ax.tricontour(Triang, u, levels = 25, colors = 'k', linewidths = 0.5, linestyles = 'solid')
    
    plt.pause(0.01)
    writer.grab_frame()
    # stop
    
    
    ##########################################################################################
    
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
    # MESH = pde.mesh(p_new,MESH.e,MESH.t,np.empty(0),MESH.regions_2d,MESH.regions_1d)
    
    # MESH.p[points_rotor,:] = (R(a1*rt)@MESH.p[points_rotor,:].T).T
    
writer.finish()
