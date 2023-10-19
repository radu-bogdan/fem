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
# Parameters
##########################################################################################

ORDER = 1
refinements = 2
plot = 0
rot_speed = 1
rots = 1

linear = '*air,*magnet,shaft_iron,*coil'
nonlinear = 'stator_iron,rotor_iron'
rotor = 'rotor_iron,*magnet,rotor_air,shaft_iron'
    
motor_npz = np.load('../meshes/motor_pizza_gap.npz', allow_pickle = True)

geoOCC = motor_npz['geoOCC'].tolist()
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

import ngsolve as ng
geoOCCmesh = geoOCC.GenerateMesh()
ngsolvemesh = ng.Mesh(geoOCCmesh)
ngsolvemesh.Refine()
ngsolvemesh.Refine()
# ngsolvemesh.Refine()

for m in range(refinements):
    
    MESH = pde.mesh.netgen(ngsolvemesh.ngmesh)
    
    from findPoints import *
    
    # ident_points, ident_edges = makeIdentifications_nogap(MESH)
    ident_points_gap, ident_edges_gap = getPoints(MESH)
    ident_points, ident_edges, jumps = makeIdentifications(MESH)
    
    
    EdgeDirection = (-1)*np.r_[np.sign(ident_points[1:jumps[1],:]-ident_points[:jumps[0],:]),
                               np.sign(ident_points[jumps[1]+1:,:]-ident_points[jumps[1]:-1,:])]
    
    
    EdgeDirectionGap = (-1)*np.sign(ident_points_gap[1:,:].astype(int)-ident_points_gap[:-1,:].astype(int))
    
    ##########################################################################################
    # Order configuration
    ##########################################################################################
    
    if ORDER == 1:
        print('Order is ', ORDER)
        space_Vh = 'N0'
        space_Qh = 'P0'
        
        order_HH = 2
        order_AA = 0
        
        order_H = 1
        order_A = 0
        HA = np.zeros(MESH.NoEdges + MESH.nt)
        
    if ORDER == 1.5:
        print('Order is ', ORDER)
        space_Vh = 'NC1'
        space_Qh = 'P0'
        
        order_HH = 2
        order_AA = 0
        
        order_H = 1
        order_A = 0
        HA = np.zeros(2*MESH.NoEdges + MESH.nt)
        
    if ORDER == 2:
        print('Order is ', ORDER)
        space_Vh = 'N1'
        space_Qh = 'P1'
        
        order_HH = 4
        order_AA = 2
        
        order_H = 2
        order_A = 1
        HA = np.zeros(2*MESH.NoEdges + 2*MESH.nt + 3*MESH.nt)
    ############################################################################################
    
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
        M_L2 = phi_L2 @ D_order_AA @ phi_L2.T
    
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
                
            if ORDER > 1:
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
        
        # if k>-1:    
        if k==0:
            H = 1e-8+np.zeros(sH)
            H = RS.T@chol(RS@RS.T).solve_A(RS@H)
            A = 0+np.zeros(sA)
        if k>0:
            # H = RS.T@chol(RS@RS.T).solve_A(RS@H)
            # A = 0+np.zeros(sA)
            Mxx = phix_Hcurl @ D_order_HH @ phix_Hcurl.T
            Myy = phiy_Hcurl @ D_order_HH @ phiy_Hcurl.T
    
            M = Mxx + Myy
            
            S= bmat([[RS@M@RS.T,RS@C.T],\
                     [C@RS.T,None]]).tocsc()
            r = np.r_[RS@M@RS.T@RS@H,C@H]
            
            
            jj = sps.linalg.spsolve(S,r)
            s = jj[:RS.shape[0]]
            H = RS.T@s
    
        HA = np.r_[H,A]
    
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
                
                if J(allHu,Hu)-J(allH,H) <= alpha*mu*(gsu@w) + np.abs(J(allH,H))*float_eps: 
                    break
                else: 
                    alpha = alpha*factor_residual
                
            print('Line search took ', time.monotonic()-tm)
            
            tm = time.monotonic()
            
            HA = HA + alpha*w_RS
            H = HA[:sH]; A = HA[sH:]
            Hx = phix_Hcurl.T@H; Hy = phiy_Hcurl.T@H
            allH = g_nonlinear_all(Hx,Hy)
            
            print('Re-evaluating H took ', time.monotonic()-tm)
            
            # print('norm cu a: ',  gs(allH,A,H))
            
            print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(allH,H)+"|| ||grad||: %2e" %np.linalg.norm(gs(allH,A,H))+"||alpha: %2e" % (alpha));
            
            if(np.linalg.norm(gs(allH,A,H)) < eps_newton): 
                break
    
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
        
        if plot == 1:
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
    
    if (m!=refinements-1):
        A_old = A
        # MESH_old_EdgesToVertices = MESH.EdgesToVertices.copy()
        ngsolvemesh.ngmesh.Refine()
        
        
    
    if (m==refinements-1):  
        A_old_newmesh = np.r_[A_old,np.c_[A_old,A_old,A_old].flatten()]
        # if ORDER == 1:
        #     u_old_newmesh = np.r_[u_old,1/2*u_old[MESH_old_EdgesToVertices[:,0]]+1/2*u_old[MESH_old_EdgesToVertices[:,1]]]
        # if ORDER == 2:
        #     u_old_newmesh = np.r_[u_old,1/2*u_old[MESH.EdgesToVertices[:,0]]+1/2*u_old[MESH.EdgesToVertices[:,1]]]
    
    if plot == 1:
        writer.finish()
    
    
err = np.sqrt((A-A_old_newmesh)@(M_L2)@(A-A_old_newmesh))
print(err)