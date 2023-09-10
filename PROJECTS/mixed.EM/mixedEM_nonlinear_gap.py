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

metadata = dict(title = 'Motor')
writer = FFMpegWriter(fps = 50, metadata = metadata)

##########################################################################################
# Loading mesh
##########################################################################################

ORDER = 1

motor_npz = np.load('../meshes/motor_pizza_gap.npz', allow_pickle = True)

geoOCC = motor_npz['geoOCC'].tolist()
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

import ngsolve as ng
geoOCCmesh = geoOCC.GenerateMesh()
ngsolve_mesh = ng.Mesh(geoOCCmesh)
# ngsolve_mesh.Refine()
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

##########################################################################################
# Order configuration
##########################################################################################
if ORDER == 1:
    polyH = 'N0'
    polyA = 'P0'
    order_HH = 2
    order_AA = 0
    u = np.zeros(MESH.NoEdges)
    
if ORDER == 2:
    poly = 'P2'
    dxpoly = 'P1'
    order_phiphi = 4
    order_dphidphi = 2
    u = np.zeros(MESH.np + MESH.NoEdges)
############################################################################################

# TODO: war grad dabei die ordnungen zu setzen


rot_speed = 1
rots = 305

tor = np.zeros(rots)
energy = np.zeros(rots)

for k in range(rots):
    
    print('Step : ', k)

    ##########################################################################################
    # Assembling stuff
    ##########################################################################################
    
    space_Vh = 'N0'
    space_Qh = 'P0'
    int_order = 4
    
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
    Ja = 0*Ja; J0 = 0*J0
    
    M0 = 0; M1 = 0; M00 = 0; M10 = 0
    for i in range(16):
        M0 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
        M1 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
        
        M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
        M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
    
    aM = phix_Hcurl@ D(int_order) @(M0) +\
         phiy_Hcurl@ D(int_order) @(M1)
    
    aJ = phi_L2(int_order)@ D(int_order) @Ja
    
    ##########################################################################################
    
    R_out, R_int = pde.hcurl.assembleR(MESH, space = 'N0', edges = 'stator_outer,left,right,airR,airL')
    
    R_L, R_LR = pde.hcurl.assembleR(MESH, space = 'N0', edges = 'left', listDOF = ident_edges[:,0])
    R_R, R_RR = pde.hcurl.assembleR(MESH, space = 'N0', edges = 'right', listDOF = ident_edges[:,1])
    
    R_AL, R_ALR = pde.hcurl.assembleR(MESH, space = 'N0', edges = 'airL', listDOF = np.roll(ident_edges_gap[:,0], -k*rot_speed))
    R_AR, R_ARR = pde.hcurl.assembleR(MESH, space = 'N0', edges = 'airR', listDOF = ident_edges_gap[:,1])
    
    if k>0:
        R_AL[-k*rot_speed+1:,:] = -R_AL[-k*rot_speed+1:,:]
        R_AL[0,:] = -R_AL[0,:]
    
    ##########################################################################################
    
    from scipy.sparse import bmat
    RS =  bmat([[R_int], [R_L-R_R], [R_AL+R_AR]])
    
    # SYS = bmat([[Mh2,C.T],\
    #             [C,None]]).tocsc()
    # rhs = np.r_[aM,np.zeros(MESH.nt)]
    
    # SYS2= bmat([[RS@Mh2@RS.T,RS@C.T],\
    #             [C@RS.T,None]]).tocsc()
    
    # rhs2= np.r_[RS@aM,np.zeros(MESH.nt)]
    
    # # tm = time.monotonic(); x = sps.linalg.spsolve(SYS,rhs); print('mixed: ',time.monotonic()-tm)
    # tm = time.monotonic(); x2 = sps.linalg.spsolve(SYS2,rhs2); print('mixed: ',time.monotonic()-tm)
    
    # A = x2[-MESH.nt:]
    # H = RS.T@x2[:-MESH.nt]
    
    ##########################################################################################
    # Solving with Newton
    ##########################################################################################
    
    from nonlinLaws import *

    sH = phix_d_Hcurl.shape[0]
    sA = phi_L2_o1.shape[0]
    sL = KK.shape[0]
    
    
    mu0 = (4*np.pi)/10**7
    H = 1e-2+np.zeros(sH)
    A = 0+np.zeros(sA)
    L = 0+np.zeros(sL)

    HAL = np.r_[H,A,L]
    
    
    def gs_hybrid(allH,A,H,L):
        gx_H_l  = allH[1]; gy_H_l  = allH[2];
        gx_H_nl = allH[8]; gy_H_nl = allH[9];
        
        r1 = phix_d_Hcurl @ D_int_order @ (gx_H_l*fem_linear + gx_H_nl*fem_nonlinear + mu0*M0) +\
             phiy_d_Hcurl @ D_int_order @ (gy_H_l*fem_linear + gy_H_nl*fem_nonlinear + mu0*M1) + Cd.T@A +KK.T@L
        r2 = Cd@H
        r3 = KK@H
        
        # r = -KK@iMd@r1 + KK@iMd@Cd.T@iBBd@(Cd@iMd@r1-r2)
        
        return np.r_[r1,r2,r3]

    def J(allH,H):
        g_H_l = allH[0]; g_H_nl = allH[7];
        return np.ones(D_int_order.size)@ D_int_order @(g_H_l*fem_linear + g_H_nl*fem_nonlinear) + mu0*aMd@H


    maxIter = 100
    epsangle = 1e-5;

    angleCondition = np.zeros(5)
    eps_newton = 1e-12
    factor_residual = 1/2
    mu = 0.0001

    tm1 = time.monotonic()
    for i in range(maxIter):
        
        H = HAL[:sH]
        A = HAL[sH:sH+sA]
        L = HAL[sH+sA:]
        
        ##########################################################################################
        Hx = phix_d_Hcurl.T@H; Hy = phiy_d_Hcurl.T@H
        
        tm = time.monotonic()
        allH = g_nonlinear_all(Hx,Hy)
        
        print('Evaluating nonlinearity took ', time.monotonic()-tm)
        
        tm = time.monotonic()
        gxx_H_l  = allH[3];  gxy_H_l  = allH[4];  gyx_H_l  = allH[5];  gyy_H_l  = allH[6];
        gxx_H_nl = allH[10]; gxy_H_nl = allH[11]; gyx_H_nl = allH[12]; gyy_H_nl = allH[13];
        
        gxx_H_Mxx = phix_d_Hcurl @ D_int_order @ sps.diags(gxx_H_nl*fem_nonlinear + gxx_H_l*fem_linear)@ phix_d_Hcurl.T
        gyy_H_Myy = phiy_d_Hcurl @ D_int_order @ sps.diags(gyy_H_nl*fem_nonlinear + gyy_H_l*fem_linear)@ phiy_d_Hcurl.T
        gxy_H_Mxy = phiy_d_Hcurl @ D_int_order @ sps.diags(gxy_H_nl*fem_nonlinear + gxy_H_l*fem_linear)@ phix_d_Hcurl.T
        gyx_H_Myx = phix_d_Hcurl @ D_int_order @ sps.diags(gyx_H_nl*fem_nonlinear + gyx_H_l*fem_linear)@ phiy_d_Hcurl.T
        
        Md = gxx_H_Mxx + gyy_H_Myy + gxy_H_Mxy + gyx_H_Myx
        
        gx_H_l  = allH[1]; gy_H_l  = allH[2]; gx_H_nl = allH[8]; gy_H_nl = allH[9];
        
        r1 = phix_d_Hcurl @ D_int_order @ (gx_H_l*fem_linear + gx_H_nl*fem_nonlinear + mu0*M0) +\
             phiy_d_Hcurl @ D_int_order @ (gy_H_l*fem_linear + gy_H_nl*fem_nonlinear + mu0*M1) + Cd.T@A + KK.T@L
        r2 = Cd@H
        r3 = KK@H
        
        print('Assembling took ', time.monotonic()-tm)
        
        
        # print(sps.linalg.eigs(Md,which = 'LR',k=2)[0])
        
        tm = time.monotonic()
        iMd = inv(Md)
        iBBd = inv(Cd@iMd@Cd.T)
        print('Inverting took ', time.monotonic()-tm)
        
        
        
        tm = time.monotonic()
        gssuR = -KK@iMd@KK.T + KK@iMd@Cd.T@iBBd@Cd@iMd@KK.T
        gsuR = -(KK@iMd@r1-r3) + KK@iMd@Cd.T@iBBd@(Cd@iMd@r1-r2)
        print('Multiplication took ', time.monotonic()-tm)
        
        
        
        tm = time.monotonic()
        wL = chol(-gssuR).solve_A(gsuR)
        print('Solving the system took ', time.monotonic()-tm)
        
        # tm = time.monotonic()
        # wL = sps.linalg.spsolve(-gssuR,gsuR)
        # print('Solving the system took ', time.monotonic()-tm)
        
        wA = iBBd@Cd@iMd@(-r1-KK.T@wL)+iBBd@r2
        wH = iMd@(-Cd.T@wA-KK.T@wL-r1)
        w = np.r_[wH,wA,wL]
        
        gsu = gs_hybrid(allH,A,H,L)
        
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
            
            HALu = HAL + alpha*w
            Hu = HALu[:sH]
            
            Hxu = phix_d_Hcurl.T@Hu; Hyu = phiy_d_Hcurl.T@Hu;
            allHu = g_nonlinear_all(Hxu,Hyu)
            
            # print(J(allHu,Hu),J(allH,H),J(allHu,Hu)-J(allH,H))
            
            if J(allHu,Hu)-J(allH,H) <= alpha*mu*(gsu@w) + np.abs(J(allH,H))*float_eps: break
            else: alpha = alpha*factor_residual
            
        print('Line search took ', time.monotonic()-tm)
        
        HAL = HALu; H = Hu; Hx = Hxu; Hy = Hyu; allH = allHu
        
        print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(allH,H)+"|| ||grad||: %2e" %np.linalg.norm(gs_hybrid(allH,A,H,L),np.inf)+"||alpha: %2e" % (alpha)); print("\n")
        if(np.linalg.norm(gs_hybrid(allH,A,H,L),np.inf) < eps_newton): break

    elapsed = time.monotonic()-tm1
    print('Solving took ', elapsed, 'seconds')
    
    ##########################################################################################
    
    Hx = phix_Hcurl.T@H; Hy = phiy_Hcurl.T@H
    
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
