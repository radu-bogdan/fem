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
from sksparse.cholmod import cholesky as chol

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
# cmap = matplotlib.colors.ListedColormap("limegreen")
cmap = plt.cm.jet

metadata = dict(title = 'Motor')
writer = FFMpegWriter(fps = 50, metadata = metadata)

##########################################################################################
# Loading mesh
##########################################################################################

ORDER = 2

motor_npz = np.load('../meshes/motor_pizza_gap.npz', allow_pickle = True)

geoOCC = motor_npz['geoOCC'].tolist()
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

import ngsolve as ng
geoOCCmesh = geoOCC.GenerateMesh()
ngsolve_mesh = ng.Mesh(geoOCCmesh)
ngsolve_mesh.Refine()
# ngsolve_mesh.Refine()

MESH = pde.mesh.netgen(ngsolve_mesh.ngmesh)

linear = '*air,*magnet,shaft_iron,*coil'
nonlinear = 'stator_iron,rotor_iron'
rotor = 'rotor_iron,*magnet,rotor_air,shaft_iron'

##########################################################################################

def makeIdentifications_nogap(MESH):

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

    edgecoord0 = np.zeros(edges0.shape[0],dtype=int)
    edgecoord1 = np.zeros(edges1.shape[0],dtype=int)

    for i in range(edges0.shape[0]):
        edgecoord0[i] = np.where(np.all(MESH.EdgesToVertices[:,:2]==edges0[i,:],axis=1))[0][0]
        edgecoord1[i] = np.where(np.all(MESH.EdgesToVertices[:,:2]==edges1[i,:],axis=1))[0][0]

    identification = np.c_[np.r_[a[ind0,0]-1,MESH.np + edgecoord0],
                            np.r_[a[ind0,1]-1,MESH.np + edgecoord1]]
    ident_points = np.c_[a[ind0,0]-1,
                          a[ind0,1]-1]
    ident_edges = np.c_[edgecoord0,
                        edgecoord1]
    return ident_points, ident_edges

# ident_points, ident_edges = makeIdentifications_nogap(MESH)

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
    
    r_sliding = 78.8354999*10**(-3)
    r_sliding2 = 79.03874999*10**(-3)
    
    i0 = np.where(np.abs(c0-r_sliding**2 )<1e-10)[0][0]
    i1 = np.where(np.abs(c0-r_sliding2**2)<1e-10)[0][0]
    
    
    ind0 = np.argsort(c0)
    
    jumps = np.r_[np.where(ind0==i0)[0][0],np.where(ind0==i1)[0][0]]
    
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

    return ident_points, ident_edges, jumps

ident_points, ident_edges, jumps = makeIdentifications(MESH)


EdgeDirection = (-1)*np.r_[np.sign(ident_points[1:jumps[1],:]-ident_points[:jumps[0],:]),
                           np.sign(ident_points[jumps[1]+1:,:]-ident_points[jumps[1]:-1,:])]


EdgeDirectionGap = (-1)*np.sign(ident_points_gap[1:,:].astype(int)-ident_points_gap[:-1,:].astype(int))

##########################################################################################
# Order configuration
##########################################################################################

if ORDER == 1:
    space_Vh = 'N0'
    space_Qh = 'P0'
    
    order_HH = 2
    order_AA = 0
    
    order_H = 1
    order_A = 0
    HA = np.zeros(MESH.NoEdges + MESH.nt)
    
if ORDER == 1.5:
    space_Vh = 'NC1'
    space_Qh = 'P0'
    
    order_HH = 2
    order_AA = 0
    
    order_H = 1
    order_A = 0
    HA = np.zeros(2*MESH.NoEdges + MESH.nt)
    
if ORDER == 2:
    space_Vh = 'N1'
    space_Qh = 'P1'
    
    order_HH = 4
    order_AA = 2
    
    order_H = 2
    order_A = 1
    HA = np.zeros(2*MESH.NoEdges + 2*MESH.nt + 3*MESH.nt)
############################################################################################

rot_speed = 10
rots = 10

tor = np.zeros(rots)
energy = np.zeros(rots)

for k in range(rots):
    
    print("\n")
    print('Step : ', k)

    ##########################################################################################
    # Assembling stuff
    ##########################################################################################
    
    tm = time.monotonic()
    
    phix_Hcurl, phiy_Hcurl = pde.hcurl.assemble(MESH, space = space_Vh, matrix = 'phi', order = order_HH)
    curlphi_Hcurl = pde.hcurl.assemble(MESH, space = space_Vh, matrix = 'curlphi', order = order_AA)
    phi_L2 = pde.l2.assemble(MESH, space = space_Qh, matrix = 'M', order = order_AA)
    
    D_order_HH = pde.int.assemble(MESH, order = order_HH)
    D_order_AA = pde.int.assemble(MESH, order = order_AA)
    
    C = phi_L2 @ D_order_AA @ curlphi_Hcurl.T

    fem_linear = pde.int.evaluate(MESH, order = order_HH, regions = linear).diagonal()
    fem_nonlinear = pde.int.evaluate(MESH, order = order_HH, regions = nonlinear).diagonal()
    fem_rotor = pde.int.evaluate(MESH, order = order_HH, regions = rotor).diagonal()
    fem_air_gap_rotor = pde.int.evaluate(MESH, order = order_HH, regions = 'air_gap_rotor').diagonal()
    
    Ja = 0; J0 = 0
    for i in range(48):
        Ja += pde.int.evaluate(MESH, order = order_AA, coeff = lambda x,y : j3[i], regions ='coil'+str(i+1)).diagonal()
        J0 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : j3[i], regions = 'coil'+str(i+1)).diagonal()
    Ja = 0*Ja; J0 = 0*J0
    
    M0 = 0; M1 = 0; M00 = 0; M10 = 0
    for i in range(16):
        M0 += pde.int.evaluate(MESH, order = order_HH, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
        M1 += pde.int.evaluate(MESH, order = order_HH, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
        
        M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
        M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
    
    aM = phix_Hcurl@ D_order_HH @(M0) +\
         phiy_Hcurl@ D_order_HH @(M1)
    
    aJ = phi_L2 @ D_order_AA @ Ja
    
    ##########################################################################################
    
    R_out, R_int = pde.hcurl.assembleR(MESH, space = space_Vh, edges = 'stator_outer,left,right,airR,airL')
    
    if ORDER == 1:
        ind_per_0 = ident_edges[:,0]
        ind_per_1 = ident_edges[:,1]
        
        ident_edges_gap_0_rolled = np.roll(ident_edges_gap[:,0], -k*rot_speed)
        
        EdgeDirectionGap_rolled = np.roll(EdgeDirectionGap[:,0], -k*rot_speed)
        EdgeDirectionGap_1 = EdgeDirectionGap[:,1]
        
        ind_gap_0 = ident_edges_gap_0_rolled
        ind_gap_1 = ident_edges_gap[:,1]
        
    if ORDER > 1:
        
        ind_per_0 = np.c_[2*ident_edges[:,0]   -1/2*(EdgeDirection[:,0]-1),
                          2*ident_edges[:,0]+1 +1/2*(EdgeDirection[:,0]-1)].ravel()
        
        ind_per_1 = np.c_[2*ident_edges[:,1]   -1/2*(EdgeDirection[:,1]-1),
                          2*ident_edges[:,1]+1 +1/2*(EdgeDirection[:,1]-1)].ravel()
        
        
        ident_edges_gap_0_rolled = np.roll(ident_edges_gap[:,0], -k*rot_speed)
        EdgeDirectionGap_rolled = np.roll(EdgeDirectionGap[:,0], -k*rot_speed)
        
        EdgeDirectionGap_1 = np.repeat(EdgeDirectionGap[:,1],2)
        
        
        ind_gap_0 = np.c_[2*ident_edges_gap_0_rolled   -1/2*(EdgeDirectionGap_rolled-1),
                          2*ident_edges_gap_0_rolled+1 +1/2*(EdgeDirectionGap_rolled-1)].ravel()
        
        ind_gap_1 = np.c_[2*ident_edges_gap[:,1]   -1/2*(EdgeDirectionGap[:,1]-1),
                          2*ident_edges_gap[:,1]+1 +1/2*(EdgeDirectionGap[:,1]-1)].ravel()
        
        EdgeDirectionGap_rolled = np.roll(np.repeat(EdgeDirectionGap[:,0],2), -2*k*rot_speed)
        
        
    R_L, R_LR = pde.hcurl.assembleR(MESH, space = space_Vh, edges = 'left', listDOF = ind_per_0)
    R_R, R_RR = pde.hcurl.assembleR(MESH, space = space_Vh, edges = 'right', listDOF = ind_per_1)
    
    R_AL, R_ALR = pde.hcurl.assembleR(MESH, space = space_Vh, edges = 'airL', listDOF = ind_gap_0); R_AL.data = EdgeDirectionGap_rolled[R_AL.indices]
    R_AR, R_ARR = pde.hcurl.assembleR(MESH, space = space_Vh, edges = 'airR', listDOF = ind_gap_1); R_AR.data = EdgeDirectionGap_1[R_AR.indices]
    
    if k>0:
        if ORDER == 1:
            R_AL[-k*rot_speed:,:] = -R_AL[-k*rot_speed:,:]
            
        if ORDER == 2:
            R_AL[-2*k*rot_speed:,:] = -R_AL[-2*k*rot_speed:,:]
    
    from scipy.sparse import bmat
    RS =  bmat([[R_int], [R_L-R_R], [R_AL+R_AR]])
    
    ##########################################################################################
    # Solving with Newton
    ##########################################################################################
    
    from nonlinLaws import *

    sH = phix_Hcurl.shape[0]
    sA = phi_L2.shape[0]
    
    
    mu0 = (4*np.pi)/10**7
    
    # if k>-1:    
    if k>=0:
        H = 1e-2+np.zeros(sH)
        H = RS.T@chol(RS@RS.T).solve_A(RS@H)
        A = 0+np.zeros(sA)
    if k<0:
        H = RS.T@chol(RS@RS.T).solve_A(RS@H)
        A = 0+np.zeros(sA)
        
        stop

    HA = np.r_[H,A]
    
    def gss(allH):
        gxx_H_l  = allH[3];  gxy_H_l  = allH[4];  gyx_H_l  = allH[5];  gyy_H_l  = allH[6];
        gxx_H_nl = allH[10]; gxy_H_nl = allH[11]; gyx_H_nl = allH[12]; gyy_H_nl = allH[13];
        
        gxx_H_Mxx = phix_Hcurl @ D_order_HH @ sps.diags(gxx_H_nl*fem_nonlinear + gxx_H_l*fem_linear)@ phix_Hcurl.T
        gyy_H_Myy = phiy_Hcurl @ D_order_HH @ sps.diags(gyy_H_nl*fem_nonlinear + gyy_H_l*fem_linear)@ phiy_Hcurl.T
        gxy_H_Mxy = phiy_Hcurl @ D_order_HH @ sps.diags(gxy_H_nl*fem_nonlinear + gxy_H_l*fem_linear)@ phix_Hcurl.T
        gyx_H_Myx = phix_Hcurl @ D_order_HH @ sps.diags(gyx_H_nl*fem_nonlinear + gyx_H_l*fem_linear)@ phiy_Hcurl.T
        
        M = gxx_H_Mxx + gyy_H_Myy + gxy_H_Mxy + gyx_H_Myx
        
        # S = bmat([[M,C.T],\
        #           [C,None]]).tocsc()
        
        S= bmat([[RS@M@RS.T,RS@C.T],\
                 [C@RS.T,None]]).tocsc()
        
        return S
    
    def gs(allH,A,H):
        gx_H_l  = allH[1]; gy_H_l  = allH[2];
        gx_H_nl = allH[8]; gy_H_nl = allH[9];
        
        r1 = phix_Hcurl @ D_order_HH @ (gx_H_l*fem_linear + gx_H_nl*fem_nonlinear + mu0*M0) +\
             phiy_Hcurl @ D_order_HH @ (gy_H_l*fem_linear + gy_H_nl*fem_nonlinear + mu0*M1) + C.T@A
             
        r2 = C@H
        return np.r_[RS@r1,r2]

    def J(allH,H):
        g_H_l = allH[0]; g_H_nl = allH[7];
        return np.ones(D_order_HH.size)@ D_order_HH @(g_H_l*fem_linear + g_H_nl*fem_nonlinear) + mu0*aM@H


    maxIter = 100
    epsangle = 1e-5;

    angleCondition = np.zeros(5)
    eps_newton = 1e-12
    factor_residual = 1/2
    mu = 0.0001

    tm1 = time.monotonic()
    for i in range(maxIter):
        
        H = HA[:sH]
        A = HA[sH:]
        
        ##########################################################################################

        Hx = phix_Hcurl.T@H; Hy = phiy_Hcurl.T@H    

        tm = time.monotonic()
        allH = g_nonlinear_all(Hx,Hy)
        gsu = gs(allH,A,H)
        gssu = gss(allH)
        
        print('Evaluating nonlinearity took ', time.monotonic()-tm)
        
        tm = time.monotonic()
        w = sps.linalg.spsolve(gssu,-gsu)
        print('Solving the system took ', time.monotonic()-tm)
        
        # w = np.r_[RS.T@w[:RS.shape[0]],
        #           w[RS.shape[0]:]]
        
        norm_w = np.linalg.norm(w)
        norm_gsu = np.linalg.norm(gsu)
        
        ##########################################################################################
        
        
        
        norm_w = np.linalg.norm(w)
        norm_gsu = np.linalg.norm(gsu)
        
        if (-(w@gsu)/(norm_w*norm_gsu)<epsangle):
            angleCondition[i%5] = 1
            if np.product(angleCondition)>0:
                w = -gsu
                print("STEP IN NEGATIVE GRADIENT DIRECTION")
                break
        else: angleCondition[i%5]=0
        
        alpha = 1
        
        # ResidualLineSearch
        # for k in range(1000):
            
        #     HALu = HAL + alpha*w
        #     Hu = HALu[:sH]
        #     Au = HALu[sH:sH+sA]
        #     Lu = HALu[sH+sA:]
            
        #     Hxu = phix_d_Hcurl.T@(Hu)
        #     Hyu = phiy_d_Hcurl.T@(Hu)
            
        #     allHu = g_nonlinear_all(Hxu,Hyu)
            
        #     if np.linalg.norm(gs_hybrid(allHu,Au,Hu,Lu)) <= np.linalg.norm(gs_hybrid(allH,A,H,L)): break
        #     else: alpha = alpha*factor_residual
        
        # AmijoBacktracking
        
        tm = time.monotonic()
        float_eps = 1e-8 #np.finfo(float).eps
        for kk in range(1000):
            
            w_RS = np.r_[RS.T@w[:RS.shape[0]], w[RS.shape[0]:]]
            
            HAu = HA + alpha*w_RS
            Hu = HAu[:sH]; Au = HAu[sH:]
            Hxu = phix_Hcurl.T@(Hu); Hyu = phiy_Hcurl.T@(Hu);
            allHu = g_nonlinear_all(Hxu,Hyu)
            
            if J(allHu,Hu)-J(allH,H) <= alpha*mu*(gsu@w) + np.abs(J(allH,H))*float_eps: break
            else: alpha = alpha*factor_residual
            
        print('Line search took ', time.monotonic()-tm)
        
        tm = time.monotonic()
        
        HA = HA + alpha*w_RS
        H = HA[:sH]; A = HA[sH:]
        Hx = phix_Hcurl.T@H; Hy = phiy_Hcurl.T@H
        allH = g_nonlinear_all(Hx,Hy)
        
        print('Re-evaluating H took ', time.monotonic()-tm)
        
        
        print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(allH,H)+"|| ||grad||: %2e" %np.linalg.norm(gs(allH,A,H))+"||alpha: %2e" % (alpha));
        if(np.linalg.norm(gs(allH,A,H)) < eps_newton): break

    elapsed = time.monotonic()-tm1
    print('Solving took ', elapsed, 'seconds')
    
    ##########################################################################################
    
    phix_Hcurl_o1, phiy_Hcurl_o1 = pde.hcurl.assemble(MESH, space = space_Vh, matrix = 'phi', order = 1)
    
    Hx = phix_Hcurl_o1.T@H; Hy = phiy_Hcurl_o1.T@H
    allH = g_nonlinear_all(Hx,Hy)
    gx_H_l  = allH[1]; gy_H_l  = allH[2];
    gx_H_nl = allH[8]; gy_H_nl = allH[9];

    fem_linear = pde.int.evaluate(MESH, order = 1, regions = linear).diagonal()
    fem_nonlinear = pde.int.evaluate(MESH, order = 1, regions = nonlinear).diagonal()

    Bx = (gx_H_l*fem_linear + gx_H_nl*fem_nonlinear)
    By = (gy_H_l*fem_linear + gy_H_nl*fem_nonlinear)
    
    if k == 0:
        fig = plt.figure()
        writer.setup(fig, "writer_test.mp4", 500)
        fig.show()
        # ax1 = fig.add_subplot(111)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    
    tm = time.monotonic()
    ax1.cla()
    ax1.set_aspect(aspect = 'equal')
    MESH.pdesurf2(A, ax = ax1)
    # MESH.pdemesh2(ax = ax)
    MESH.pdegeom(ax = ax1)
    Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
    # ax1.tricontour(Triang, u[:MESH.np], levels = 25, colors = 'k', linewidths = 0.5, linestyles = 'solid')
    
    ax2.cla()
    ax2.set_aspect(aspect = 'equal')
    MESH.pdesurf2(Bx**2+By**2, ax = ax2)
    # MESH.pdemesh2(ax = ax)
    MESH.pdegeom(ax = ax2)
    Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
    
    writer.grab_frame()
    
    ##########################################################################################
    
    if k != rots-1:
        
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

    
writer.finish()

p0 = MESH.p[:,0]
p1 = MESH.p[:,1]
t = MESH.t[:,0:3]
# nt = MESH.nt

p0d = np.c_[p0[t[:,0]],p0[t[:,1]],p0[t[:,2]]].ravel()
p1d = np.c_[p1[t[:,0]],p1[t[:,1]],p1[t[:,2]]].ravel()
# td = np.r_[:3*MESH.nt].reshape(MESH.nt,3)
# Triang = matplotlib.tri.Triangulation(p0d, p1d, td)

# plt.tripcolor(Triang, Hx**2+Hy**2, cmap = plt.cm.jet, lw = 0.1)



# plt.tripcolor(Triang.x, Triang.y, Triang.x, shading='gouraud' )
# plt.tripcolor(p0d, p1d, Triang.x, shading='gouraud' )

# f = lambda x,y: x**2+y**2

# f = f(p0d,p1d)

# MESH.pdesurf2(f)