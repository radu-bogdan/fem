import ngsolve as ngs
from netgen.webgui import Draw as DrawGeo
from ngsolve.webgui import Draw
from ngsolve.krylovspace import CGSolver
from ttictoc import tic, toc
import numpy as np

ngs.SetHeapSize(1000000000)

doarr = np.r_[1,1,1]

#################################################################################################################################################
nonlinlaw = 3
deg = 3
#################################################################################################################################################

mesh = ngs.Mesh('whatever.vol')
mesh.Curve(2)
# mesh.Refine()

print("using 3d mesh with ne=", mesh.ne, " elements and nv=", mesh.nv, " points")

deg = deg + 1

HcurlCoil = ngs.HCurl(mesh, order = deg, dirichlet = "coil_outer|coil_inner|coil_up|coil_down", definedon = 'coil', nograds = True)
TotalCurrent = 1000
coil_depth = 0.1

n = ngs.specialcf.normal(mesh.dim)
gfuDirichlet = ngs.GridFunction(HcurlCoil)
g = ngs.CF([(0,0,TotalCurrent/coil_depth) if bc=="coil_inner" else (0,0,0) for bc in mesh.GetBoundaries()])
gfuDirichlet.Set(g,ngs.BND)

# Weak form
T, Tstar = HcurlCoil.TnT()
Kt = ngs.BilinearForm(ngs.curl(Tstar)*ngs.curl(T)*ngs.dx)
    
# Assembly
with ngs.TaskManager(): Kt.Assemble()

r = - Kt.mat * gfuDirichlet.vec

# Solving
maxres = 1e-5
maxit = 100000
                              
Tsol = ngs.GridFunction(HcurlCoil)
# print("Solving...")

with ngs.TaskManager():
    iterativeSolver = CGSolver(Kt.mat, freedofs = HcurlCoil.FreeDofs(), atol  = maxres,  maxiter = maxit)
    Tsol.vec.data = iterativeSolver * r + gfuDirichlet.vec

# if len(iterativeSolver.residuals) == maxit: print("... Failure!")
# else: print("... Success!")
print(f"Number of iterations = {iterativeSolver.iterations}/{maxit} | Residual = {iterativeSolver.residuals[-1]}")

# Draw(ngs.curl(Tsol), mesh, vectors = { "grid_size" : 150},clipping = {"x" : 0, "y" : 0, "z" : -1, "dist" : 0})
with ngs.TaskManager():
    flux_bd = ngs.Integrate(ngs.curl(Tsol) * n, mesh, definedon = mesh.Boundaries("coil_cut_2"))
# print(flux_bd)

# Hcurl = ngs.HCurl(mesh, order = p, nograds = True)
Hcurl = ngs.HCurl(mesh, order = deg, nograds = True, dirichlet = 'ambient_face')
u,v = Hcurl.TnT()

K = ngs.BilinearForm(ngs.curl(u)*ngs.curl(v)*ngs.dx)
c = ngs.Preconditioner(K, type = "local")

with ngs.TaskManager(): K.Assemble()

f = ngs.LinearForm(Hcurl)
f +=  ngs.curl(Tsol)* ngs.curl(v) * ngs.dx

with ngs.TaskManager(): f.Assemble()

Hs = ngs.GridFunction(Hcurl)
# print("Solving...")

with ngs.TaskManager():
    iterativeSolver = CGSolver(K.mat, c.mat, atol  = maxres,  maxiter = maxit)
    # iterativeSolver = CGSolver(K.mat, freedofs = Hcurl.FreeDofs(), tol  = maxres,  maxiter = maxit)
    Hs.vec.data = iterativeSolver * f.vec

# if len(iterativeSolver.residuals) == maxit: print("... Failure!")
# else: print("... Success!")
print(f"Number of iterations = {iterativeSolver.iterations}/{maxit} | Residual = {iterativeSolver.residuals[-1]}")

# Draw(ngs.curl(Hs), mesh, vectors = { "grid_size" : 150},clipping = {"x" : 0, "y" : 0, "z" : -1, "dist" : 0})

with ngs.TaskManager():
    flux_bd = ngs.Integrate(ngs.curl(Hs) * n, mesh, definedon = mesh.Boundaries("coil_cut_2"))
    



deg = deg - 1

    
if doarr[0]==1:

    #################################################################################################################################################
    print("\nVECTOR POTENTIAL FORMULATION\n")
    #################################################################################################################################################

        
    from bhdata import BHCurves
    fun_dw  = BHCurves(nonlinlaw)
    fun_w   = fun_dw.Integrate()
    fun_ddw = fun_dw.Differentiate()

    # mu0 = 1.256636e-6
    mu0 = 2e-6
    nu0 = 1/mu0

    linear = "coil|ambient|default"
    nonlinear = "r_steel|l_steel|mid_steel"

    lin = 0

    #################################################################################################################################################
        
    # print(HCurl.ndof)
    # print(mesh.GetMaterials())

    HCurl = ngs.HCurl(mesh, order = deg, nograds = True, dirichlet = 'ambient_face')
    # HCurl = ngs.HCurl(mesh, order = deg, nograds = True)
    u,v = HCurl.TnT()


    # Nonlinear:

    maxit = 100000
    tol2 = 1e-8
    regb = 1e-12

    A = ngs.GridFunction(HCurl)
    B = ngs.curl(A)
    normB = ngs.sqrt(B*B + regb)


    if lin == 1: cf_energy = mesh.MaterialCF({linear: nu0/2*B*B, nonlinear: nu0/2*B*B}, default = nu0/2*B*B).Compile()
    else: cf_energy = mesh.MaterialCF({linear: nu0/2*B*B, nonlinear: fun_w(normB)}, default = nu0/2*B*B).Compile()

    def fun_W():
        with ngs.TaskManager(): res = ngs.Integrate(cf_energy - ngs.curl(A)*Hs, mesh, order = 2*deg)
        # with ngs.TaskManager(): res = ngs.Integrate(cf_energy - ngs.curl(Hs)*A, mesh)
        return res

    ir = ngs.IntegrationRule(ngs.fem.ET.TET, order = 3*deg)


    if lin == 1: cf_rhs = mesh.MaterialCF({linear: nu0, nonlinear: nu0}, default = nu0).Compile()
    else: cf_rhs = mesh.MaterialCF({linear: nu0, nonlinear: fun_dw(normB)/normB}, default = nu0).Compile()

    rhs = ngs.LinearForm(HCurl)
    rhs += ngs.SymbolicLFI(cf_rhs*B*ngs.curl(v) - ngs.curl(v)*Hs, intrule = ir)

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

    if lin == 1: cf_iter = mesh.MaterialCF({linear: nu0*Id, nonlinear: nu0*Id}, default = nu0*Id).Compile()
    else: cf_iter = mesh.MaterialCF({linear: nu0*Id, nonlinear: fun1*Id + fun2*BBt}, default = nu0*Id).Compile()

    K_iter = ngs.BilinearForm(HCurl)
    K_iter += ngs.SymbolicBFI(cf_iter*ngs.curl(u)*ngs.curl(v), intrule = ir)
    # K_iter += (cf_iter*ngs.curl(u)*ngs.curl(v))*ngs.dx
    C_iter = ngs.Preconditioner(K_iter, type = "local")

    def fun_ddW():
        with ngs.TaskManager(): K_iter.Assemble()
        return K_iter

        
    #################################################################################################################################################
        
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
        # du.vec.data = da.mat.Inverse(HCurl.FreeDofs(), inverse="sparsecholesky") * dw.vec 
        # iterativeSolver = CGSolver(K_iter.mat, freedofs = HCurl.FreeDofs(), atol = 1e-2,  maxiter = maxit, printrates = False)
        with ngs.TaskManager():
            iterativeSolver = CGSolver(K_iter.mat, pre = C_iter.mat, tol  = 1e-8,  maxiter = maxit)
            du.vec.data = iterativeSolver * dw.vec
        
        if len(iterativeSolver.residuals) == maxit: print("... Failure! MaxIt Reached!")
        # print(f"Number of iterations = {iterativeSolver.iterations}/{maxit} | Residual = {iterativeSolver.residuals[-1]}")
        tm2 = toc()

        nrm = ngs.InnerProduct(du.vec,dw.vec)
        
        if it == 1:
            nrm0 = nrm
        
        # wn = 1e12
        # if abs(wo-w) < tol2:
        # if abs(wn-w) < tol2:
        if nrm/nrm0 < tol2:
            print("converged to desired tolerance")
            break
        elif nrm/nrm0 < 1e-2*tol2:
            # print("converged to desired tolerance due to relres")
        #     print("stopped early due to stagnation")
            break
        else:
            # linesearch
            uo.vec.data = A.vec.data
            wo = w
            alpha = 1
            for init in range(1,21):
                A.vec.data -= alpha*du.vec.data
                wn = fun_W()
                if wn < w - alpha*0.1*nrm:
                    print("Iter: %2d | assem : %.2fs | CG took %.2fs with %4d iterations | alpha : %.2f | energy = %.10f | relres = %.2e |"  %(it,tm1,tm2,iterativeSolver.iterations,alpha,w,nrm/nrm0))
                    break
                else:
                    alpha = alpha/2
                    A.vec.data = uo.vec.data
                        
    #################################################################################################################################################














if doarr[1]==1:
    #################################################################################################################################################
    print("\nSCALAR POTENTIAL FORMULATION\n")
    #################################################################################################################################################

    from bhdata import BHCurves
    fun_dw  = BHCurves(-nonlinlaw)
    fun_w   = fun_dw.Integrate()
    fun_ddw = fun_dw.Differentiate()

    lin = 0

    #################################################################################################################################################


    # print(HCurl.ndof)
    # print(mesh.GetMaterials())

    H1 = ngs.H1(mesh, order = deg)
    # H1 = ngs.H1(mesh, order = deg, dirichlet = 'ambient_face')
    u,v = H1.TnT()


    # Nonlinear:
    maxit = 100000
    tol2 = 1e-8
    regh = 1e-12

    psi = ngs.GridFunction(H1)
    H = ngs.grad(psi) + Hs
    normH = ngs.sqrt(H*H + regh)

    if lin == 1: cf_coenergy = mesh.MaterialCF({linear: mu0/2*H*H, nonlinear: mu0/2*H*H}, default = mu0/2*H*H).Compile()
    else: cf_coenergy = mesh.MaterialCF({linear: mu0/2*H*H, nonlinear: fun_w(normH)}, default = mu0/2*H*H).Compile()

    def fun_W():
        # with ngs.TaskManager(): 
        res = ngs.Integrate(cf_coenergy, mesh)
        return res

    ir = ngs.IntegrationRule(ngs.fem.ET.TET, order = 3*deg)

    if lin == 1: cf_rhs = mesh.MaterialCF({linear: mu0, nonlinear: mu0}, default = mu0).Compile()
    else: cf_rhs = mesh.MaterialCF({linear: mu0, nonlinear: fun_dw(normH)/normH}, default = mu0).Compile()


    rhs = ngs.LinearForm(H1)
    rhs += ngs.SymbolicLFI(cf_rhs*H*ngs.grad(v), intrule = ir)
    # rhs = ngs.LinearForm(cf_rhs*H*ngs.grad(v)*ngs.dx)


    def fun_dW(): #implicitly depending on A!
        # with ngs.TaskManager(): 
        rhs.Assemble()
        return rhs


    Id = ngs.CF((1,0,0,
                 0,1,0,
                 0,0,1), dims=(3,3))

    HHt = ngs.CF((H[0]*H[0], H[0]*H[1], H[0]*H[2],
                H[1]*H[0], H[1]*H[1], H[1]*H[2],
                H[2]*H[0], H[2]*H[1], H[2]*H[2]), dims=(3,3))


    fun1 = fun_dw(normH)/normH
    fun2 = (fun_ddw(normH) - fun_dw(normH)/normH)/(normH*normH)

    if lin == 1: cf_iter = mesh.MaterialCF({linear: mu0*Id, nonlinear: mu0*Id}, default = mu0*Id).Compile()
    else: cf_iter = mesh.MaterialCF({linear: mu0*Id, nonlinear: fun1*Id + fun2*HHt}, default = mu0*Id).Compile()

    K_iter = ngs.BilinearForm(H1)
    K_iter += ngs.SymbolicBFI(cf_iter*ngs.grad(u)*ngs.grad(v), intrule = ir)
    # K_iter += ((cf_iter*ngs.grad(u))*ngs.grad(v))*ngs.dx

    C_iter = ngs.Preconditioner(K_iter, type = "local")

    def fun_ddW():
        # with ngs.TaskManager(): 
        K_iter.Assemble()
        return K_iter

    #################################################################################################################################################

    with ngs.TaskManager():
        print("Using 3D mesh with ne=", mesh.ne, "elements and nv=", mesh.nv, "points and " ,H1.ndof, "DOFs.\n ")

        with ngs.TaskManager(): psi.Set(ngs.CF((0)))

        du = ngs.GridFunction(H1)
        uo = ngs.GridFunction(H1)
        wo = 1e12

        for it in range(1,maxit+1):
            
            tic()
            # w  = fun_W()
            res = ngs.Integrate(cf_coenergy, mesh)
            w = res
            tm10 = toc()

            tic()
            # dw = fun_dW()
            rhs.Assemble()
            dw = rhs
            tm11 = toc()

            tic()
            # da = fun_ddW()
            K_iter.Assemble()
            da = K_iter
            tm12 = toc()
            
            tic()
            # iterativeSolver = CGSolver(K_iter.mat, freedofs = HCurl.FreeDofs(), atol = 1e-2,  maxiter = maxit, printrates = False)
            with ngs.TaskManager():
                iterativeSolver = CGSolver(K_iter.mat, pre = C_iter.mat, tol  = 1e-4,  maxiter = maxit)
                # iterativeSolver = CGSolver(K_iter.mat, freedofs = H1.FreeDofs(), tol  = 1e-2,  maxiter = maxit)
                du.vec.data = iterativeSolver * dw.vec
                # du.vec.data = da.mat.Inverse(H1.FreeDofs(), inverse="sparsecholesky") * dw.vec 
            
            if len(iterativeSolver.residuals) == maxit: print("... Failure! MaxIt Reached!")
            # print(f"Number of iterations = {iterativeSolver.iterations}/{maxit} | Residual = {iterativeSolver.residuals[-1]}")
            tm2 = toc()

            nrm = ngs.InnerProduct(du.vec,dw.vec)
            
            if it == 1:
                nrm0 = nrm

            wn = 1e12
            if nrm/nrm0 < tol2:
            # if abs(wo-w) < tol2:
            # if abs(wn-w) < tol2:
                print("converged to desired tolerance")
                break
            elif abs(wo-w) < tol2*1e-2:
                print("stopped early due to stagnation")
                break
            else:
                # linesearch
                uo.vec.data = psi.vec.data
                wo = w
                alpha = 1
                for init in range(1,21):
                    psi.vec.data -= alpha*du.vec.data
                    wn = fun_W()
                    if wn < w - alpha*0.1*nrm:
                        print("Iter: %2d | assem : %.2fs,%.2fs,%.2fs | CG took %.2fs with %4d iterations | alpha : %.2f | energy = %.10f | relres = %.2e |"  %(it,tm10,tm11,tm12,tm2,iterativeSolver.iterations,alpha,w,nrm/nrm0))
                        break
                    else:
                        alpha = alpha/2
                        psi.vec.data = uo.vec.data
                        
    #################################################################################################################################################













if doarr[2]==1:

    #################################################################################################################################################
    print("\nREDUCED MIXED FORMULATION\n")
    #################################################################################################################################################

    from bhdata import BHCurves
    fun_dw  = BHCurves(nonlinlaw)
    fun_w   = fun_dw.Integrate()
    fun_ddw = fun_dw.Differentiate()

    lin = 0

    #################################################################################################################################################
        
    Q = ngs.VectorL2(mesh, order = deg-1)
    V = ngs.H1(mesh, order = deg)
    # V = ngs.H1(mesh, order = deg, dirichlet = "ambient_face")

    X = Q*V
    (p,u), (q,v) = X.TnT()

    # Nonlinear:

    # maxit = 2000

    tol2 = 1e-8
    regb = 1e-12

    B_psi = ngs.GridFunction(X)

    B = B_psi.components[0]
    psi = B_psi.components[1]

    normB = ngs.sqrt(B*B + regb)

    if lin == 1: cf_energy = mesh.MaterialCF({linear: nu0/2*B*B, nonlinear: nu0/2*B*B}, default = nu0/2*B*B).Compile()
    cf_energy = mesh.MaterialCF({linear: nu0/2*B*B, nonlinear: fun_w(normB)}, default = nu0/2*B*B).Compile()

    def fun_W():
        with ngs.TaskManager(): res = ngs.Integrate(cf_energy - Hs*B, mesh, order = 3*deg)
        return res

    ir = ngs.IntegrationRule(ngs.fem.ET.TET, order = 3*deg)

    if lin == 1: cf_rhs = mesh.MaterialCF({linear: nu0, nonlinear: nu0}, default = nu0).Compile()
    else: cf_rhs = mesh.MaterialCF({linear: nu0, nonlinear: fun_dw(normB)/normB}, default = nu0).Compile()

    rhs = ngs.LinearForm(X)
    rhs += ngs.SymbolicLFI(cf_rhs*B*q -ngs.grad(psi)*q -Hs*q +B*ngs.grad(v), intrule = ir)
    # rhs = ngs.LinearForm((cf_rhs*B*q -ngs.grad(psi)*q -Hs*q +B*ngs.grad(v))*ngs.dx)

    def fun_dW():
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

    if lin == 1: cf_iter = mesh.MaterialCF({linear: nu0*Id, nonlinear: nu0*Id}, default = nu0*Id).Compile()
    else: cf_iter = mesh.MaterialCF({linear: nu0*Id, nonlinear: fun1*Id + fun2*BBt}, default = nu0*Id).Compile()

    # K_iter = ngs.BilinearForm(X)
    K_iter = ngs.BilinearForm(X, condense = True)
    K_iter += ngs.SymbolicBFI(cf_iter*p*q - ngs.grad(u)*q + p*ngs.grad(v), intrule = ir)
    # K_iter += (cf_iter*p*q - ngs.grad(u)*q + p*ngs.grad(v))*ngs.dx
    C_iter = ngs.Preconditioner(K_iter, type = "local")

    def fun_ddW():
        with ngs.TaskManager(): K_iter.Assemble()
        return K_iter

    #################################################################################################################################################

    print("Using 3D mesh with ne=", mesh.ne, "elements and nv=", mesh.nv, "points and " ,X.ndof, "DOFs.\n ")

    with ngs.TaskManager(): B.Set(ngs.CF((0,0,0)))
    with ngs.TaskManager(): psi.Set(ngs.CF((0)))

    du = ngs.GridFunction(X)
    uo = ngs.GridFunction(X)
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

            # iterativeSolver = CGSolver(mat = da.mat, pre = C_iter.mat, tol = 1e-4,  maxiter = maxit)
            # # du.vec.data = iterativeSolver * dw.vec

            # du.vec.data = iterativeSolver * rhs_mod
            # # du.vec.data = RT * du.vec

            # du.vec.data += da.harmonic_extension * du.vec
            # du.vec.data += da.inner_solve * rhs.vec
            # du.vec.data = da.mat.Inverse(X.FreeDofs())*dw.vec

            rhs_mod = (da.harmonic_extension_trans * dw.vec).Evaluate()
            iterativeSolver = CGSolver(mat = da.mat, pre = C_iter.mat, tol  = 1e-8,  maxiter = maxit)
            # iterativeSolver = CGSolver(mat = da.mat, freedofs = X.FreeDofs(coupling = True), tol  = 1e-4,  maxiter = maxit)
            
            # rows,cols,vals = da.mat.COO()
            # import scipy.sparse as sp
            # A = sp.csr_matrix((vals,(rows,cols)))
            # print(A.shape,A.max(),A.min())

            du.vec.data = iterativeSolver * rhs_mod
            # print(ngs.InnerProduct(du.vec,du.vec))
            du.vec.data += da.harmonic_extension * du.vec
            # print(ngs.InnerProduct(du.vec,du.vec))
            du.vec.data += da.inner_solve * rhs.vec
            # print(ngs.InnerProduct(du.vec,du.vec))

        
        if len(iterativeSolver.residuals) == maxit: print("... reached maxit!")
        # print(f"Number of iterations = {iterativeSolver.iterations}/{maxit} | Residual = {iterativeSolver.residuals[-1]}")
        tm2 = toc()

        nrm = ngs.InnerProduct(du.vec,dw.vec)
        # print(nrm)
        
        if it == 1:
            nrm0 = nrm

        # wn = 1e12
        # if abs(wo-w) < tol2:
        if nrm/nrm0 < tol2:
            # print(wo)
            # print(w)
            print("converged to desired tolerance")
            break
        elif abs(wo-w) < tol2*1e-2:
            print("stopped early due to stagnation")
            break
        else:
            # linesearch
            uo.vec.data = B_psi.vec.data
            wo = w
            alpha = 1
            for init in range(1,21):
                B_psi.vec.data -= alpha*du.vec.data
                wn = fun_W()
                if wn < w - alpha*0.1*nrm:
                    print("Iter: %2d | assem : %.2fs | CG took %.2fs with %4d iterations | alpha : %.2f | energy = %.10f | relres = %.2e |"  %(it,tm1,tm2,iterativeSolver.iterations,alpha,w,nrm/(nrm0+1e-2)))
                    break
                else:
                    alpha = alpha/2
                    B_psi.vec.data = uo.vec.data

    #################################################################################################################################################