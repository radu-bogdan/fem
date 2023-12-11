from imports import *
from imports import np,pde,sps

writer = FFMpegWriter(fps = 50, metadata = dict(title = 'Motor'))

##########################################################################################
# Parameters
##########################################################################################

ORDER = 1
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

# geoOCC = motor_npz['geoOCC'].tolist()
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

if len(sys.argv) > 1:
    level = int(sys.argv[1])
else:
    level = 2
    
    
print("LEVEL " , level)

for m in range(refinements):
    
    open_file = open('mesh_full'+str(level)+'.pkl', "rb")
    # open_file = open('mesh'+str(level)+'.pkl', "rb")
    MESH = dill.load(open_file)[0]
    open_file.close()
    
    from findPoints import *
    
    tm = time.monotonic()
    # getPoints(MESH)
    # ident_points_gap = getPointsNoEdges(MESH)
    print('getPoints took  ', time.monotonic()-tm)
    
    tm = time.monotonic()
    # makeIdentifications(MESH)
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
        psi = np.zeros(MESH.np)
        b = np.zeros(2*MESH.nt)
        
    if ORDER == 2:
        poly = 'P2'
        dxpoly = 'P1'
        order_phiphi = 4
        order_dphidphi = 2
        psi = np.zeros(MESH.np + MESH.NoEdges)
        b = np.zeros(6*MESH.nt)
    ############################################################################################
    
    
    
    ############################################################################################
    ## Brauer/Nonlinear laws ... ?
    ############################################################################################
    
    sys.path.insert(1,'../mixed.EM')
    # from nonlinLaws import *
    from nonlinLaws_brauer_fit import *
    # from nonlinLaws_bosch import *
                           
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
        dphix_H1_order_phiphi, dphiy_H1_order_phiphi = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_phiphi)
        phi_L2 = pde.l2.assemble(MESH, space = dxpoly, matrix = 'M', order = order_dphidphi)
        
        # RS = getRS_H1_nonzero(MESH,ORDER,poly,k,rot_speed)
        R_out, R_int = pde.h1.assembleR(MESH, space = poly, edges = 'stator_outer')
        RS = bmat([[R_int[1:]],[R_out]])
        
        
        ##########################################################################################
            
        D_order_dphidphi = pde.int.assemble(MESH, order = order_dphidphi)
        D_order_phiphi = pde.int.assemble(MESH, order = order_phiphi)
        
        MASS = phi_H1 @ D_order_phiphi @ phi_H1.T
        Kxx = dphix_H1 @ D_order_dphidphi @ dphix_H1.T
        Kyy = dphiy_H1 @ D_order_dphidphi @ dphiy_H1.T
        Cx = phi_L2 @ D_order_dphidphi @ dphix_H1.T
        Cy = phi_L2 @ D_order_dphidphi @ dphiy_H1.T
    
        fem_linear = pde.int.evaluate(MESH, order = order_dphidphi, regions = linear).diagonal()
        fem_nonlinear = pde.int.evaluate(MESH, order = order_dphidphi, regions = nonlinear).diagonal()
                
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
        
        aM = dphix_H1_order_phiphi@ D_order_phiphi @(M0) +\
             dphiy_H1_order_phiphi@ D_order_phiphi @(M1)
        
        aMnew = aM
        
        
        # fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1), M00, u_height=0)
        # fig.show()
        
        # print('Assembling + stuff ', time.monotonic()-tm)
        ##########################################################################################
        
        
        ##########################################################################################
        # Importing Hj
        ##########################################################################################
        
        from mixedEM_linear_gap import H as Hj
        # Hj = 0*Hj
        phix_Hcurl, phiy_Hcurl = pde.hcurl.assemble(MESH, space = 'N0', matrix = 'phi', order = order_dphidphi)
        Hjx = phix_Hcurl.T@Hj
        Hjy = phiy_Hcurl.T@Hj
        
        ##########################################################################################
        # Solving with Newton
        ##########################################################################################
        
        sb = 2*(phi_L2.shape[0])
        spsi = phi_H1.shape[0]
        
        maxIter = 100
        epsangle = 1e-5
        
        angleCondition = np.zeros(5)
        eps_newton = 1e-8
        factor_residual = 1/2
        mu = 0.0001
        
        def gss(b):
            bx = b[:len(b)//2]; by = b[len(b)//2:]
            
            fxx_u_Kxx = phi_L2 @ D_order_dphidphi @ sps.diags(fxx_linear(bx,by)*fem_linear + fxx_nonlinear(bx,by)*fem_nonlinear)@ phi_L2.T
            fyy_u_Kyy = phi_L2 @ D_order_dphidphi @ sps.diags(fyy_linear(bx,by)*fem_linear + fyy_nonlinear(bx,by)*fem_nonlinear)@ phi_L2.T
            fxy_u_Kxy = phi_L2 @ D_order_dphidphi @ sps.diags(fxy_linear(bx,by)*fem_linear + fxy_nonlinear(bx,by)*fem_nonlinear)@ phi_L2.T
            fyx_u_Kyx = phi_L2 @ D_order_dphidphi @ sps.diags(fyx_linear(bx,by)*fem_linear + fyx_nonlinear(bx,by)*fem_nonlinear)@ phi_L2.T
            
            R = bmat([[fxx_u_Kxx,fxy_u_Kxy],
                      [fyx_u_Kyx,fyy_u_Kyy]])
            
            C = bmat([[Cx],[Cy]])
            
            return R,C
            
        def gs(b,psi):
            bx = b[:len(b)//2]; by = b[len(b)//2:]
            
            r1 =  np.r_[phi_L2 @ D_order_dphidphi @ (fx_linear(bx,by)*fem_linear + fx_nonlinear(bx,by)*fem_nonlinear -Hjx +M0_dphi) - Cx @ psi,
                        phi_L2 @ D_order_dphidphi @ (fy_linear(bx,by)*fem_linear + fy_nonlinear(bx,by)*fem_nonlinear -Hjy +M1_dphi) - Cy @ psi]
            
            r2 = dphix_H1 @ D_order_dphidphi @ bx +\
                 dphiy_H1 @ D_order_dphidphi @ by
            
            return r1,r2
        
        # def J(psi):
        #     psix = dphix_H1.T@psi; psiy = dphiy_H1.T@psi
        #     return np.ones(D_order_dphidphi.size)@ D_order_dphidphi @(g_linear(Hjx +psix,Hjy +psiy)*fem_linear + g_nonlinear(Hjx +psix,Hjy +psiy)*fem_nonlinear)
        
        def J(b,psi):
            bx = b[:len(b)//2]; by = b[len(b)//2:]
            psix = dphix_H1.T@psi; psiy = dphiy_H1.T@psi
            return np.ones(D_order_dphidphi.size)@ D_order_dphidphi @(f_linear(bx,by)*fem_linear + f_nonlinear(bx,by)*fem_nonlinear +(-Hjx+M0_dphi)*bx +(-Hjy+M1_dphi)*by -psix*bx -psiy*by)        
        
        
        tm = time.monotonic()
        
        print('Initial guess compute took  ', time.monotonic()-tm)
            # stop
            
        tm2 = time.monotonic()
        for i in range(maxIter):
            
            R,C = gss(b)
            
            r1,r2 = gs(b,psi)
            
            A = bmat([[R,-C@RS.T],
                      [RS@C.T,None]]).tocsc()
            
            r = np.r_[r1,0*RS@r2]
            
            # tm = time.monotonic()
            # # wS = chol(AA).solve_A(-rr)
            # w = sps.linalg.spsolve(A,-r)
            # wb = w[:sb]
            # wpsi = RS.T@w[sb:]
            # print('Solving took ', time.monotonic()-tm)
            
            
            tm = time.monotonic()
            iR = pde.tools.fastBlockInverse(R)
            AA = RS@C.T@iR@C@RS.T
            rr = RS@(-C.T@iR@r1+0*r2)
            wpsi = RS.T@chol(AA).solve_A(-rr)
            wb = iR@(C@wpsi-r1)
            w = np.r_[RS@wpsi,wb]
            print('Solving took ', time.monotonic()-tm)
            
            # print('dif:',np.linalg.norm(wb-wb2),np.linalg.norm(wpsi-wpsi2))
            
            # print("RESIDUAL SHIT: ", r@w,J(psi),J(psi+wpsi),np.linalg.norm(RS@C.T@b,np.inf))
            
            
            # MESH.pdesurf2(wpsi,cbar=1)
            # stop
            
            # w = wS
            
            # wpsi = RS.T@wS
            
            # MESH.pdesurf2(wpsi)
            
            # wb = iR@(C@wpsi+r1)
            
            # stop
            
            alpha = 1
            
            # ResidualLineSearch
            # for k in range(1000):
            #     print("RESIDUAL: ", J2(psi+alpha*wpsi),J2(psi),J2(psi+alpha*wpsi)-J2(psi),alpha*mu*(rhs@(RS@wpsi)),alpha*mu*(r@w))
            #     if np.linalg.norm(np.r_[gs(b+alpha*wb,psi+alpha*wpsi)],np.inf) <= np.linalg.norm(np.r_[gs(b,psi)],np.inf): break
            #     else: alpha = alpha*factor_residual
                
            # print("RESIDUAL: ", J2(psi+alpha*wpsi),J2(psi),J2(psi+alpha*wpsi)-J2(psi),alpha*mu*(rhs@(RS@wpsi)),alpha*mu*(r@w))
            
            # AmijoBacktracking
            float_eps = 1e-12; #float_eps = np.finfo(float).eps
            for kk in range(1000):
                if J(b+alpha*wb,psi+alpha*wpsi)-J(b,psi) <= alpha*mu*(r@w)+ np.abs(J(b,psi))*float_eps: break
                else: alpha = alpha*factor_residual
            
            b_old_i = b
            psi_old_i = psi
            
            b = b + alpha*wb
            psi = psi + alpha*wpsi
            
            print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(b,psi)+"|| ||grad||: %2e" %np.linalg.norm(r,np.inf)+"||alpha: %2e" % (alpha)+"|| J(u) : %2e" %J(b,psi))
                        
            # if ( np.linalg.norm(r,np.inf) < eps_newton):
            #     break
            if (np.abs(J(b,psi)-J(b_old_i,psi_old_i)) < 1e-5):
                break
            
        # print('im out!!!!!!!!!')
            
        elapsed = time.monotonic()-tm2
        print('Solving took ', elapsed, 'seconds')
        
        # stop
        
        # ax1.cla()
        dphix_H1_o1, dphiy_H1_o1 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = 0)
        psix = dphix_H1_o1.T@psi; psiy = dphiy_H1_o1.T@psi
        
        Hx = Hjx + psix
        Hy = Hjy + psiy
        
        allH = g_nonlinear_all(Hx,Hy)
        gx_H_l  = allH[1]; gy_H_l  = allH[2];
        gx_H_nl = allH[8]; gy_H_nl = allH[9];
    
        fem_linear = pde.int.evaluate(MESH, order = 0, regions = linear).diagonal()
        fem_nonlinear = pde.int.evaluate(MESH, order = 0, regions = nonlinear).diagonal()
    
        Bx = (gx_H_l*fem_linear + gx_H_nl*fem_nonlinear) + 1/nu0*M00
        By = (gy_H_l*fem_linear + gy_H_nl*fem_nonlinear) + 1/nu0*M10
        
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
                # writer.setup(fig, "writer_test.mp4", 500)
                fig.show()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                # ax3 = fig.add_subplot(223)
                # ax4 = fig.add_subplot(224)
            
            tm = time.monotonic()
            ax1.cla()
            ax1.set_aspect(aspect = 'equal')
            MESH.pdesurf2(psi[:MESH.np], ax = ax1, cbar=1)
            # MESH.pdemesh2(ax = ax)
            MESH.pdegeom(ax = ax1)
            # Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
            # ax1.tricontour(Triang, u[:MESH.np], levels = 25, colors = 'k', linewidths = 0.5, linestyles = 'solid')
            
            ax2.cla()
            ax2.set_aspect(aspect = 'equal')
            MESH.pdesurf2((Bx)**2+(By)**2, ax = ax2)
            # MESH.pdesurf2((Hjx+ux)**2+(Hjy+uy)**2, ax = ax2)
            # MESH.pdemesh2(ax = ax)
            MESH.pdegeom(ax = ax2)
            # Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
            
            # ax3.cla()
            # # ax3.set_aspect(aspect = 'equal')
            # ax3.plot(tor)
            # ax3.plot((energy[2:]-energy[1:-1])*(ident_points_gap.shape[0]))
            
            # ax4.cla()
            # # ax3.set_aspect(aspect = 'equal')
            # ax4.plot(energy)
        
        
        
        
            print('Plotting took  ', time.monotonic()-tm)
            
            tm = time.monotonic()
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
        # ngsolvemesh.ngmesh.Refine()
        level = level + 1
    
    # if plot == 1:
        # writer.finish()

if refinements>1:
    err = np.sqrt((u-u_old_newmesh)@(MASS)@(u-u_old_newmesh))/np.sqrt((u)@(MASS)@(u))
    err2= np.sqrt((u-u_old_newmesh)@(Kxx+Kyy)@(u-u_old_newmesh))/np.sqrt((u)@(Kxx+Kyy)@(u))
    
    M_L2 = phi_L2 @ D_order_dphidphi @ phi_L2.T
    errB = np.sqrt((Bx_old_newmesh-Bx)@(M_L2)@(Bx_old_newmesh-Bx))/np.sqrt((Bx)@(M_L2)@(Bx)) + \
           np.sqrt((By_old_newmesh-By)@(M_L2)@(By_old_newmesh-By))/np.sqrt((By)@(M_L2)@(By))
    print(err,err2,errB)
    
# print('tor by energy diff ', (energy[1]-energy[0])*(ident_points_gap.shape[0]))