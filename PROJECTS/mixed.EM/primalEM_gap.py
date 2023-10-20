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
import ngsolve as ng
# cmap = matplotlib.colors.ListedColormap("limegreen")
cmap = plt.cm.jet

metadata = dict(title = 'Motor')
writer = FFMpegWriter(fps = 50, metadata = metadata)

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
refinements = 1
plot = 1
rot_speed = 1
rots = 300

linear = '*air,*magnet,shaft_iron,*coil'
nonlinear = 'stator_iron,rotor_iron'
rotor = 'rotor_iron,*magnet,rotor_air,shaft_iron'

geoOCC = motor_npz['geoOCC'].tolist()
geoOCCmesh = geoOCC.GenerateMesh()
ngsolvemesh = ng.Mesh(geoOCCmesh)
# ngsolvemesh.Refine()
# ngsolvemesh.Refine()
# ngsolvemesh.Refine()
# ngsolvemesh.Refine()

for m in range(refinements):
    
    # MESH.refinemesh()
    MESH = pde.mesh.netgen(ngsolvemesh.ngmesh)
    
    
    from findPoints import *
    
    tm = time.monotonic()
    ident_points_gap, ident_edges_gap = getPoints(MESH)
    # ident_points_gap = getPointsNoEdges(MESH)
    print('getPoints took  ', time.monotonic()-tm)
    
    tm = time.monotonic()
    ident_points, ident_edges, jumps = makeIdentifications(MESH)
    print('makeIdentifications took  ', time.monotonic()-tm)
    
    
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
    
    tor = np.zeros(rots)
    energy = np.zeros(rots)
    
    for k in range(rots):
        
        print('\n Step : ', k)
        
        ##########################################################################################
        # Assembling stuff
        ##########################################################################################
        
        # if ORDER == 1:
        #     u = np.zeros(MESH.np)
        # if ORDER == 2:
        #     u = np.zeros(MESH.np + MESH.NoEdges)
       
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
       
        # manual stuff: (removing the point in the three corners...)
        corners = np.r_[0,jumps,ident_points.shape[0]-1]
        ind1 = np.setdiff1d(np.r_[0:R_L.shape[0]], corners)
        R_L = R_L[ind1,:]
        
        corners = np.r_[0,jumps,ident_points.shape[0]-1]
        ind1 = np.setdiff1d(np.r_[0:R_R.shape[0]], corners)
        R_R = R_R[ind1,:]
        
        
        
        ident0 = np.roll(ident_points_gap[:,0], -k*rot_speed)
        ident1 = ident_points_gap[:,1]
        
        R_AL, R_ALR = pde.h1.assembleR(MESH, space = poly, edges = 'airL', listDOF = ident0)
        R_AR, R_ARR = pde.h1.assembleR(MESH, space = poly, edges = 'airR', listDOF = ident1)
            
        if k>0:
            R_AL[-k*rot_speed:,:] = -R_AL[-k*rot_speed:,:]
            
        if ORDER == 2:
            
            R_AL2, _ = pde.h1.assembleR(MESH, space = poly, edges = 'airL', listDOF = MESH.np + np.roll(ident_edges_gap[:,0], -k*rot_speed))
            R_AR2, _ = pde.h1.assembleR(MESH, space = poly, edges = 'airR', listDOF = MESH.np + ident_edges_gap[:,1])
            
            if k>0:
                R_AL2[-k*rot_speed:,:] = -R_AL2[-k*rot_speed:,:] # old
                
            from scipy.sparse import bmat
            R_AL =  bmat([[R_AL], [R_AL2]])
            R_AR =  bmat([[R_AR], [R_AR2]])
            
        
        from scipy.sparse import bmat
        RS =  bmat([[R_int], [R_L-R_R], [R_AL+R_AR]])
        ##########################################################################################
        
        
            
        D_order_dphidphi = pde.int.assemble(MESH, order = order_dphidphi)
        D_order_phiphi = pde.int.assemble(MESH, order = order_phiphi)
        # D_order_phiphi_b = pde.int.assembleB(MESH, order = order_phiphi)
        
        
        MASS = phi_H1 @ D_order_phiphi @ phi_H1.T
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
            
            M0_dphi += pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
            M1_dphi += pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
        
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
        
        tm = time.monotonic()
        if k>0:
            u = RS.T@chol(RS@RS.T).solve_A(RS@u)
        print('Initial guess compute took  ', time.monotonic()-tm)
            # stop
            
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
            float_eps = 1e-12; #float_eps = np.finfo(float).eps
            for kk in range(1000):
                if J(u+alpha*w)-J(u) <= alpha*mu*(gsu@wS) + np.abs(J(u))*float_eps: break
                else: alpha = alpha*factor_residual
                
            u = u + alpha*w
            
            print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(u)+"|| ||grad||: %2e" %np.linalg.norm(RS @ gs(u),np.inf)+"||alpha: %2e" % (alpha))
            
            if ( np.linalg.norm(RS @ gs(u),np.inf) < eps_newton):
                break
            
        # print('im out!!!!!!!!!')
            
        elapsed = time.monotonic()-tm2
        print('Solving took ', elapsed, 'seconds')
        
        
        # ax1.cla()
        ux = dphix_H1_o1.T@u; uy = dphiy_H1_o1.T@u
        
        # fig = MESH.pdesurf((ux-1/nu0*M1_dphi)**2+(uy+1/nu0*M0_dphi)**2, u_height = 0, cmax = 5)
        # fig.show()
        
        # fig = MESH.pdesurf_hybrid(dict(trig = 'P1',quad = 'Q0',controls = 1), u[0:MESH.np], u_height=1)
        # fig.show()
        
        
        # ax.cla()
        # MESH.pdesurf2(u[:MESH.np],ax = ax)
        
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # time.sleep(0.1)
        
        # input()
        
        if plot == 1:
            if k == 0:
                fig = plt.figure()
                writer.setup(fig, "writer_test.mp4", 500)
                fig.show()
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
            
            tm = time.monotonic()
            ax1.cla()
            ax1.set_aspect(aspect = 'equal')
            MESH.pdesurf2(u[:MESH.np], ax = ax1)
            # MESH.pdemesh2(ax = ax)
            MESH.pdegeom(ax = ax1)
            Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
            ax1.tricontour(Triang, u[:MESH.np], levels = 25, colors = 'k', linewidths = 0.5, linestyles = 'solid')
            
            ax2.cla()
            ax2.set_aspect(aspect = 'equal')
            MESH.pdesurf2(ux**2+uy**2, ax = ax2)
            # MESH.pdemesh2(ax = ax)
            MESH.pdegeom(ax = ax2)
            Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
            
            ax3.cla()
            # ax3.set_aspect(aspect = 'equal')
            ax3.plot(tor)
        
        
        
        
            print('Plotting took  ', time.monotonic()-tm)
            
            tm = time.monotonic()
            # writer.grab_frame()
            writer.grab_frame()
            print('Grabbing took  ', time.monotonic()-tm)
            # stop
        
        
        ##########################################################################################
        # Torque computation
        ##########################################################################################
        
        
        fem_linear = pde.int.evaluate(MESH, order = order_dphidphi, regions = linear).diagonal()
        fem_nonlinear = pde.int.evaluate(MESH, order = order_dphidphi, regions = nonlinear).diagonal()
        
        ux = dphix_H1.T@u; uy = dphiy_H1.T@u
        f = lambda ux,uy : f_linear(ux,uy)*fem_linear + f_nonlinear(ux,uy)*fem_nonlinear
        fx = lambda ux,uy : fx_linear(ux,uy)*fem_linear + fx_nonlinear(ux,uy)*fem_nonlinear
        fy = lambda ux,uy : fy_linear(ux,uy)*fem_linear + fy_nonlinear(ux,uy)*fem_nonlinear
        
        u_Pk = phi_H1.T@u
        
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
        
        
        v1x_fem = pde.int.evaluate(MESH, order = order_dphidphi, coeff = v1x, regions = '*air_gap').diagonal()
        v1y_fem = pde.int.evaluate(MESH, order = order_dphidphi, coeff = v1y, regions = '*air_gap').diagonal()
        v2x_fem = pde.int.evaluate(MESH, order = order_dphidphi, coeff = v2x, regions = '*air_gap').diagonal()
        v2y_fem = pde.int.evaluate(MESH, order = order_dphidphi, coeff = v2y, regions = '*air_gap').diagonal()
        
        scale_fem = pde.int.evaluate(MESH, order = order_dphidphi, coeff = scale, regions = '*air_gap,'+rotor).diagonal()
        one_fem = pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : 1+0*x+0*y, regions = '*air_gap,'+rotor).diagonal()        
        
        b1 = uy
        b2 = -ux
        fu = f(ux,uy)+1/2*(M0_dphi*uy-M1_dphi*ux)
        fbb1 =  fy(ux,uy)+M0_dphi
        fbb2 = -fx(ux,uy)+M1_dphi
        a_Pk = u_Pk
        
        
        term1 = (fu + fbb1*b1 +fbb2*b2 )*(v1x_fem + v2y_fem)
        term2 = (fbb1*b1)*v1x_fem + (fbb2*b1)*v2x_fem + (fbb1*b2)*v1y_fem + (fbb2*b2)*v2y_fem
        
        term_2 = -(term1+term2)
        tor[k] = one_fem@D_order_dphidphi@term_2
        print('Torque:', tor[k])
        
        
        ##########################################################################################
        
        rotor = 'rotor_iron,*magnet,rotor_air,shaft_iron,air_gap_rotor'
        
        fem_rotor = pde.int.evaluate(MESH, order = 0, regions = rotor).diagonal()
        trig_rotor = MESH.t[np.where(fem_rotor)[0],0:3]
        points_rotor = np.unique(trig_rotor)
    
        R = lambda x: np.array([[np.cos(x),-np.sin(x)],
                                [np.sin(x), np.cos(x)]])
        
        a1 = 2*np.pi/(ident_points_gap.shape[0]-1)/8
        
        p_new = MESH.p.copy(); t_new = MESH.t.copy()
        p_new[points_rotor,:] = (R(a1*rot_speed)@MESH.p[points_rotor,:].T).T
        
        m_new = R(a1*rot_speed)@m_new
        
        MESH = pde.mesh(p_new,MESH.e,MESH.t,np.empty(0),MESH.regions_2d,MESH.regions_1d)
        ##########################################################################################
        
        
        
    # dphix_L2, dphiy_L2 = pde.l2.assemble(MESH, space = poly, matrix = 'K', order = order_dphidphi)
    # phid_L2 = pde.l2.assemble(MESH, space = poly, matrix = 'M', order = order_phiphi)
    # Md = phid_L2 @ D_order_phiphi @ phid_L2
    # Kdxx = dphix_L2 @ D_order_dphidphi @ dphix_L2.T
    # Kdyy = dphiy_L2 @ D_order_dphidphi @ dphiy_L2.T
    
    if refinements>1:
        if (m!=refinements-1):
            # print("m is ",m)
            # ud_old = phi_H1.T@u
            u_old = u
            MESH_old_EdgesToVertices = MESH.EdgesToVertices.copy()
            # print(MESH,u.shape)
            # MESH.refinemesh()
            ngsolvemesh.ngmesh.Refine()
            # print(MESH)
        
        if (m==refinements-1):
            # ud_new = phi_H1.T@u
            # ud_old_newmesh = np.r_[ud_old,ud_old,ud_old,ud_old]
            
            if ORDER == 1:
                u_old_newmesh = np.r_[u_old,1/2*u_old[MESH_old_EdgesToVertices[:,0]]+1/2*u_old[MESH_old_EdgesToVertices[:,1]]]
            if ORDER == 2:
                u_old_newmesh = np.r_[u_old,1/2*u_old[MESH.EdgesToVertices[:,0]]+1/2*u_old[MESH.EdgesToVertices[:,1]]]
    
    if refinements == 0:
        ngsolvemesh.ngmesh.Refine()
    
    if plot == 1:
        writer.finish()

if refinements>1:
    err = np.sqrt((u-u_old_newmesh)@(Kxx+Kyy+MASS)@(u-u_old_newmesh))/np.sqrt((u)@(Kxx+Kyy+MASS)@(u))
    print(err)
