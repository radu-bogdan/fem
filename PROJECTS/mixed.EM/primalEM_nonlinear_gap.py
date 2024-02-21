from imports import *
from imports import np,pde,sps

writer = FFMpegWriter(fps = 50, metadata = dict(title = 'Motor'))

##########################################################################################
# Parameters
##########################################################################################

ORDER = 2
refinements = 1
plot = 1
# rot_speed = (((18*2-1)*2-1)*2-1)*2-1
rot_speed = 1
rots = 306#*2*2#*2
rots = 1

linear = '*air,*magnet,shaft_iron,*coil'
nonlinear = 'stator_iron,rotor_iron'
rotor = 'rotor_iron,*magnet,rotor_air,shaft_iron'

motor_npz = np.load('../meshes/data.npz', allow_pickle = True)

m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

if len(sys.argv) > 1:
    level = int(sys.argv[1])
else:
    level = 2
    
print("LEVEL " , level)



############################################################################################
## Brauer/Nonlinear laws ... ?
############################################################################################

sys.path.insert(1,'../mixed.EM')
# from nonlinLaws import *
from nonlinLaws_brauer_fit import *
# from nonlinLaws_bosch import *
                       
############################################################################################



for m in range(refinements):
    
    open_file = open('mesh_full'+str(level)+'.pkl', "rb")
    # open_file = open('mesh'+str(level)+'.pkl', "rb")
    MESH = dill.load(open_file)[0]
    open_file.close()
    
    from findPoints import *
    
    tm = time.monotonic()
    # getPoints(MESH)
    print('getPoints took  ', time.monotonic()-tm)
    
    tm = time.monotonic()
    # makeIdentifications(MESH)
    print('makeIdentifications took  ', time.monotonic()-tm)
    
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
    
    tor = np.zeros(rots)
    energy = np.zeros(rots)
    
    for k in range(rots):
        
        print('\n Step : ', k)
        
        ##########################################################################################
        # Assembling stuff
        ##########################################################################################
       
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
        
        MASS = phi_H1 @ D_order_phiphi @ phi_H1.T
        Kxx = dphix_H1 @ D_order_dphidphi @ dphix_H1.T
        Kyy = dphiy_H1 @ D_order_dphidphi @ dphiy_H1.T
        Cx = phi_L2 @ D_order_dphidphi @ dphix_H1.T
        Cy = phi_L2 @ D_order_dphidphi @ dphiy_H1.T
        
        D_stator_outer = pde.int.evaluateB(MESH, order = order_phiphi, edges = 'stator_outer')
        B_stator_outer = phi_H1b@ D_stator_outer @ phi_H1b.T
    
        fem_linear = pde.int.evaluate(MESH, order = order_dphidphi, regions = linear).diagonal()
        fem_nonlinear = pde.int.evaluate(MESH, order = order_dphidphi, regions = nonlinear).diagonal()
        
        # Identification of "freeDofs"
        R0, RSS = pde.h1.assembleR(MESH, space = poly, edges = 'stator_outer')
        
        # RS = getRS_H1(MESH,ORDER,poly,k,rot_speed)
        RS = RSS
        
        ##########################################################################################
        # Assembling J,M
        ##########################################################################################
        
        # penalty = 1e10
        
        Ja = 0; J0 = 0
        for i in range(48):
            Ja += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : j3[i], regions = 'coil'+str(i+1)).diagonal()
            J0 += pde.int.evaluate(MESH, order = order_dphidphi, coeff = lambda x,y : j3[i], regions = 'coil'+str(i+1)).diagonal()
        # Ja = 0*Ja; J0 = 0*J0
        
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
        
        tm2 = time.monotonic()
        for i in range(maxIter):
            
            gssu = RS @ gss(u) @ RS.T
            gsu = RS @ gs(u)
            
            tm = time.monotonic()
            wS = chol(gssu).solve_A(-gsu)
            # wS = sps.linalg.spsolve(gssu,-gsu)
            print('Solving took ', time.monotonic()-tm)
            
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
            
            
            # ax = fig.add_subplot(111)
            # ax.cla()
            # MESH.pdesurf2(u,ax = ax)
            # MESH.pdegeom(ax = ax)
            # MESH.pdemesh2(ax = ax)
            # plt.pause(0.01)
            # input()
            
            u_old_i = u
            u = u + alpha*w
            
            
            print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(u)+"|| ||grad||: %2e" %np.linalg.norm(RS @ gs(u),np.inf)+"||alpha: %2e" % (alpha))
            
            # if ( np.linalg.norm(RS @ gs(u),np.inf) < eps_newton):
                # break
            if (np.abs(J(u)-J(u_old_i)) < 1e-5):
                break
            
        elapsed = time.monotonic()-tm2
        print('Solving took ', elapsed, 'seconds')
        
        
        ux = dphix_H1_o1.T@u; uy = dphiy_H1_o1.T@u
        
        if plot == 1:
            if k == 0:
                fig = plt.figure()
                # writer.setup(fig, "writer_test.mp4", 500)
                fig.show()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                # ax3 = fig.add_subplot(223)
                # ax4 = fig.add_subplot(224)
            
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
            # Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
            
            # ax3.cla()
            # # ax3.set_aspect(aspect = 'equal')
            # ax3.plot(tor)
            # ax3.plot((energy[2:]-energy[1:-1])*(MESH.ident_points_gap.shape[0]))
            
            # ax4.cla()
            # # ax3.set_aspect(aspect = 'equal')
            # ax4.plot(energy)
        
        
        
        
            print('Plotting took  ', time.monotonic()-tm)
            
            tm = time.monotonic()
            writer.grab_frame()
            # writer.grab_frame()
            print('Grabbing took  ', time.monotonic()-tm)
            # stop
        
        
        ##########################################################################################
        # Magnetic energy
        ##########################################################################################
        
        energy[k] = J(u)
        
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
        
        #outer radius rotor
        r_outer = 78.63225*10**(-3)
        #sliding mesh rotor
        r_sliding = 78.8354999*10**(-3)
        #sliding mesh stator
        r_sliding2 = 79.03874999*10**(-3)
        #inner radius stator
        r_inner = 79.242*10**(-3)
        
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
        fbb1 =  fy(ux,uy)#+M0_dphi
        fbb2 = -fx(ux,uy)#+M1_dphi
        # a_Pk = u_Pk
        
        
        term1 = (fu + fbb1*b1 +fbb2*b2 )*(v1x_fem + v2y_fem)
        term2 = (fbb1*b1)*v1x_fem +\
                (fbb2*b1)*v2x_fem +\
                (fbb1*b2)*v1y_fem +\
                (fbb2*b2)*v2y_fem
        
        term_2 = (term1+term2)
        tor[k] = one_fem@D_order_dphidphi@term_2
        print('Torque:', tor[k])
        print('Torque:', energy[k])
        
        
        ##########################################################################################
        
        if k != rots-1:
            
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
        
    # MESH.pdesurf2(u)
    # MESH.pdemesh2()
    
    if refinements>1:
        if (m!=refinements-1):
            # print("m is ",m)
            # ud_old = phi_H1.T@u
            u_old = u
            
            dphix_H1, dphiy_H1 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_dphidphi)
            Bx_old = dphix_H1.T@u
            By_old = dphiy_H1.T@u
            
            
            MESH_old_EdgesToVertices = MESH.EdgesToVertices.copy()
            # print(MESH,u.shape)
            # MESH.refinemesh()
            # ngsolvemesh.ngmesh.Refine()
            level = level + 1
            # print(MESH)
        
        if (m==refinements-1):
            # ud_new = phi_H1.T@u
            # ud_old_newmesh = np.r_[ud_old,ud_old,ud_old,ud_old]
            
            if ORDER == 1:
                u_old_newmesh = np.r_[u_old,1/2*u_old[MESH_old_EdgesToVertices[:,0]]+1/2*u_old[MESH_old_EdgesToVertices[:,1]]]
            if ORDER == 2:
                u_old_newmesh = np.r_[u_old,1/2*u_old[MESH.EdgesToVertices[:,0]]+1/2*u_old[MESH.EdgesToVertices[:,1]]]
            
            dphix_H1, dphiy_H1 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_dphidphi)
            Bx = dphix_H1.T@u
            By = dphiy_H1.T@u
            
            Bx_old_newmesh = np.r_[Bx_old,np.c_[Bx_old,Bx_old,Bx_old].flatten()]
            By_old_newmesh = np.r_[By_old,np.c_[By_old,By_old,By_old].flatten()]
            
    if refinements == 0:
        level = level + 1
    
    if plot == 1:
        writer.finish()

if refinements>1:
    err = np.sqrt((u-u_old_newmesh)@(MASS)@(u-u_old_newmesh))/np.sqrt((u)@(MASS)@(u))
    err2= np.sqrt((u-u_old_newmesh)@(Kxx+Kyy)@(u-u_old_newmesh))/np.sqrt((u)@(Kxx+Kyy)@(u))
    
    M_L2 = phi_L2 @ D_order_dphidphi @ phi_L2.T
    errB = np.sqrt((Bx_old_newmesh-Bx)@(M_L2)@(Bx_old_newmesh-Bx))/np.sqrt((Bx)@(M_L2)@(Bx)) + \
           np.sqrt((By_old_newmesh-By)@(M_L2)@(By_old_newmesh-By))/np.sqrt((By)@(M_L2)@(By))
    print(err,err2,errB)
    
# print('tor by energy diff ', (energy[1]-energy[0])*(ident_points_gap.shape[0]))