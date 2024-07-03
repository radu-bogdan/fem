import ngsolve as ngs
from ttictoc import tic, toc
from ngsolve.krylovspace import CGSolver
from ngcotree import *


def solve(HCurl,A,mesh,deg,J,fun_w,fun_dw,fun_ddw,linear,nonlinear):

    # linear = "coil_plus|coil_minus"
    # nonlinear = "stator"

    mu0 = 1.256636e-6
    nu0 = 1/mu0

    # HCurl = ngs.HCurl(mesh, order = deg, nograds = True, dirichlet = 'outer_face|coil_plus_face|coil_minus_face|stator_face', autoupdate=True)
    # HCurl = ngs.HCurl(mesh, order = deg, nograds = True)
    u,v = HCurl.TnT()


    # Nonlinear:
    maxit = 1_000_000
    tol2 = 1e-13
    regb = 1e-8

    # A = ngs.GridFunction(HCurl, nested=True)
    B = ngs.curl(A)
    normB = ngs.sqrt(B*B + regb)

    ir = ngs.IntegrationRule(ngs.fem.ET.TET, order = 3*deg)

    cf_energy = mesh.MaterialCF({linear: nu0/2*B*B, nonlinear: fun_w(normB)}, default = nu0/2*B*B).Compile()

    def fun_W():
        # with ngs.TaskManager(): res = ngs.Integrate(cf_energy - J*A, mesh)
        with ngs.TaskManager(): res = ngs.Integrate(cf_energy - J*A, mesh, order = 3*deg)
        # with ngs.TaskManager(): res = ngs.Integrate(cf_energy - ngs.curl(Hs)*A, mesh)
        # print("res:" + str(ngs.Integrate(cf_energy, mesh)))
        return res


    cf_rhs = mesh.MaterialCF({linear: nu0, nonlinear: fun_dw(normB)/normB}, default = nu0).Compile()

    rhs = ngs.LinearForm(HCurl)
    rhs += ngs.SymbolicLFI(cf_rhs*B*ngs.curl(v) - J*v, intrule = ir)

    # rhs = ngs.LinearForm((cf_rhs*B*ngs.curl(v) - ngs.curl(v)*Hs)*ngs.dx)
    # rhs = ngs.LinearForm((cf_rhs*B*ngs.curl(v) - ngs.curl(Hs)*v)*ngs.dx)

    def fun_dW(): #implicitly depending on A!
        with ngs.TaskManager(): rhs.Assemble()
        return rhs

    Id = ngs.CF((1,0,0,
                 0,1,0,
                 0,0,1), dims=(3,3))

    BBt = ngs.CF((B[0]*B[0], B[0]*B[1], B[0]*B[2],
                  B[1]*B[0], B[1]*B[1], B[1]*B[2],
                  B[2]*B[0], B[2]*B[1], B[2]*B[2]), dims = (3,3))

    fun1 = fun_dw(normB)/normB
    fun2 = (fun_ddw(normB) - fun_dw(normB)/normB)/(normB*normB)

    cf_iter = mesh.MaterialCF({linear: nu0*Id, nonlinear: fun1*Id + fun2*BBt}, default = nu0*Id).Compile()

    K_iter = ngs.BilinearForm(HCurl)
    K_iter += ngs.SymbolicBFI(cf_iter*ngs.curl(u)*ngs.curl(v), intrule = ir)
    # K_iter += (cf_iter*ngs.curl(u)*ngs.curl(v))*ngs.dx
    C_iter = ngs.Preconditioner(K_iter, type = "local")

    def fun_ddW():
        with ngs.TaskManager(): K_iter.Assemble()
        return K_iter

    newFreeDofs = CoTreeBitArray(mesh, HCurl, plot = False)

    print("Using 3D mesh with ne=", mesh.ne, "elements and nv=", mesh.nv, "points and " ,HCurl.ndof, "DOFs.\n ")

    with ngs.TaskManager(): A.Set(ngs.CF((0,0,0)))

    du = ngs.GridFunction(HCurl)
    uo = ngs.GridFunction(HCurl)
    wo = 1e12

    for it in range(1,maxit+1):
        tic()
        # with ngs.TaskManager():
        #     K_iter.Assemble()
        #     rhs.Assemble()
        #     res = ngs.Integrate(cf_energy - Hs*ngs.curl(A), mesh)
        
        w  = fun_W()
        dw = fun_dW()
        da = fun_ddW()
        tm1 = toc()
        
        tic()

    
        jac = K_iter.mat.CreateSmoother(newFreeDofs)

        class SymmetricGS(ngs.BaseMatrix):
            def __init__ (self, smoother):
                super(SymmetricGS, self).__init__()
                self.smoother = smoother
            def Mult (self, x, y):
                y[:] = 0.0
                self.smoother.Smooth(y, x)
                self.smoother.SmoothBack(y,x)
            def Height (self):
                return self.smoother.height
            def Width (self):
                return self.smoother.height
            
        jacmod = SymmetricGS(jac)
        
        # iterativeSolver = CGSolver(K_iter.mat, freedofs = HCurl.FreeDofs(), atol = 1e-2,  maxiter = maxit, printrates = False)
        # iterativeSolver = CGSolver(K_iter.mat, pre = C_iter.mat, tol  = 1e-8,  maxiter = maxit*10)
        with ngs.TaskManager():
            # iterativeSolver = CGSolver(K_iter.mat, freedofs = newFreeDofs, tol  = 1e-8,  maxiter = maxit, printrates = False)
            iterativeSolver = CGSolver(K_iter.mat, pre = jacmod, tol  = 1e-13,  maxiter = maxit, printrates = False)

            du.vec.data = iterativeSolver * dw.vec
            # du.vec.data = da.mat.Inverse(newFreeDofs, inverse="sparsecholesky") * dw.vec 
        
        # print('MAXdu: ' + str(du.vec.FV().NumPy().max()))
        # print('MAXdw: ' + str(dw.vec.FV().NumPy().max()))

        if len(iterativeSolver.residuals) == maxit: print("... Failure!")
        # print(f"Number of iterations = {iterativeSolver.iterations}/{maxit} | Residual = {iterativeSolver.residuals[-1]}")
        tm2 = toc()
        
        nrm = ngs.InnerProduct(du.vec,dw.vec)
        
        if it == 1:
            nrm0 = nrm
        
        # wn = 1e12
        if abs(wo-w)/abs(w+regb) < tol2:
        # if abs(wn-w) < tol2:
        # if nrm/nrm0 < tol2:
            # print(wo,w)
            # print("converged to desired tolerance")
            break
        elif abs(wo-w) < tol2*1e-2:
            print("stopped early due to stagnation")
        #     break
        else:
            # linesearch
            # print("Doing LS:")
            uo.vec.data = A.vec.data
            wo = w
            alpha = 1
            for init in range(1,2100):
                A.vec.data -= alpha*du.vec.data
                wn = fun_W()
                # print(w,wn)
                if wn < w - alpha*0.01*nrm:
                    # print("Iter: %2d | assem : %.2fs | CG took %.2fs with %4d iterations | alpha : %.2f | energy = %.10f | relres = %.2e |"  %(it,tm1,tm2,iterativeSolver.iterations,alpha,w,nrm/nrm0))
                    print("Iter: %2d | assem : %.2fs | CG took %.2fs with %4d iterations | alpha : %.2f | energy = %.10f | relres = %.2e |"  %(it,tm1,tm2,len(iterativeSolver.residuals),alpha,w,nrm/nrm0))
                    break
                else:
                    # print(alpha)
                    alpha = alpha/2
                    A.vec.data = uo.vec.data
    
    # HDiv = ngs.HDiv(mesh, order = deg)
    # B = ngs.GridFunction(HDiv)
    # with ngs.TaskManager(): B.Set(ngs.curl(A))

    print(A.vec.data.FV().NumPy().max())

    return A,it



def solve_2d(H1,A,mesh,deg,J,fun_w,fun_dw,fun_ddw,linear,nonlinear):

    mu0 = 1.256636e-6
    nu0 = 1/mu0

    u,v = H1.TnT()

    rot = ngs.CF((0,1,-1,0), dims=(2,2))
    def curl2d(a):
        return rot*ngs.grad(a)

    
    maxit = 10_000_000
    tol2 = 1e-13
    regb = 1e-8

    B = curl2d(A)
    normB = ngs.sqrt(B*B + regb)

    ir = ngs.IntegrationRule(ngs.fem.ET.TRIG, order = 3*deg)

    cf_energy = mesh.MaterialCF({linear: nu0/2*B*B, nonlinear: fun_w(normB)}, default = nu0/2*B*B).Compile()

    def fun_W():
        # with ngs.TaskManager(): res = ngs.Integrate(cf_energy - curl2d(A)*Hs, mesh, order = 2*deg)
        with ngs.TaskManager(): res = ngs.Integrate(cf_energy - A*J, mesh, order = 3*deg)
        return res


    cf_rhs = mesh.MaterialCF({linear: nu0, nonlinear: fun_dw(normB)/normB}, default = nu0).Compile()

    rhs = ngs.LinearForm(H1)
    rhs += ngs.SymbolicLFI(cf_rhs*B*curl2d(v) - J*v, intrule = ir)

    def fun_dW(): #implicitly depending on A!
        with ngs.TaskManager(): rhs.Assemble()
        return rhs

    Id = ngs.CF((1,0,
                 0,1), dims = (2,2))

    BBt = ngs.CF((B[0]*B[0], B[0]*B[1],
                  B[1]*B[0], B[1]*B[1]), dims = (2,2))

    fun1 = fun_dw(normB)/normB
    fun2 = (fun_ddw(normB) - fun_dw(normB)/normB)/(normB*normB)

    cf_iter = mesh.MaterialCF({linear: nu0*Id, nonlinear: fun1*Id + fun2*BBt}, default = nu0*Id).Compile()

    K_iter = ngs.BilinearForm(H1)
    K_iter += ngs.SymbolicBFI(cf_iter*curl2d(u)*curl2d(v), intrule = ir)
    C_iter = ngs.Preconditioner(K_iter, type = "local")

    def fun_ddW():
        with ngs.TaskManager(): K_iter.Assemble()
        return K_iter
    
    print("Using 2D mesh with ne=", mesh.ne, "elements and nv=", mesh.nv, "points and " ,H1.ndof, "DOFs.\n ")

    with ngs.TaskManager(): A.Set(ngs.CF((0)))

    du = ngs.GridFunction(H1)
    uo = ngs.GridFunction(H1)
    wo = 1e12

    for it in range(1,maxit+1):
        tic()
        # with ngs.TaskManager():
        #     K_iter.Assemble()
        #     rhs.Assemble()
        #     res = ngs.Integrate(cf_energy - Hs*ngs.curl(A), mesh)
        
        w  = fun_W()
        dw = fun_dW()
        da = fun_ddW()
        tm1 = toc()
        
        tic()

        with ngs.TaskManager():
            du.vec.data = da.mat.Inverse(H1.FreeDofs(), inverse = "sparsecholesky") * dw.vec
            iterativeSolver = CGSolver(K_iter.mat, freedofs = H1.FreeDofs(), atol = 1e-2,  maxiter = maxit, printrates = False)
            # iterativeSolver = CGSolver(K_iter.mat, pre = C_iter.mat, tol  = 1e-8,  maxiter = maxit)
            # du.vec.data = iterativeSolver * dw.vec
        
        if len(iterativeSolver.residuals) == maxit: print("... Failure!")
        # print(f"Number of iterations = {iterativeSolver.iterations}/{maxit} | Residual = {iterativeSolver.residuals[-1]}")
        tm2 = toc()

        nrm = ngs.InnerProduct(du.vec,dw.vec)

        if it == 1:
            nrm0 = nrm
        
        # wn = 1e12
        if abs(wo-w)/abs(w+regb) < tol2:
        # if abs(wo-w)/abs(w) < tol2:
        # if abs(wn-w) < tol2:
        # if nrm/nrm0 < tol2:
            print("Iter: %2d | assem : %.2fs | CG took %.2fs | alpha : %.2f | energy = %.10f | relres = %.2e |"  %(it,tm1,tm2,alpha,w,nrm/nrm0))
            # print("converged to desired tolerance")
            break
        elif abs(wo-w)< tol2*1e-2:
        #     # print(abs(wo-w),abs(w),alpha,nrm,nrm0)
            print("stopped early due to stagnation | energy= %.10f" %w)
        #     # print("Iter: %2d | assem : %.2fs | CG took %.2fs with %4d iterations | alpha : %.2f | energy = %.10f | relres = %.2e |"  %(it,tm1,tm2,iterativeSolver.iterations,alpha,w,nrm/nrm0))
        #     break
        else:
            # linesearch
            uo.vec.data = A.vec.data
            wo = w
            alpha = 1
            for init in range(1,2100):
                A.vec.data -= alpha*du.vec.data
                wn = fun_W()
                # if wn < w - alpha*1/2*nrm:
                if wn < w - alpha*0.01*nrm:
                    print("Iter: %2d | assem : %.2fs | CG took %.2fs | alpha : %.2f | energy = %.10f | relres = %.2e |"  %(it,tm1,tm2,alpha,w,nrm/nrm0))
                    break
                else:
                    alpha = alpha/2
                    A.vec.data = uo.vec.data
            # print(alpha)
    return A,it