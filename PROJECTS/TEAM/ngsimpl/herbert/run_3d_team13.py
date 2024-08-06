import ngsolve as ngs
from nonlinear import do
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt
from copy import deepcopy
import pyngcore as ngcore
ngcore.SetNumThreads(48)


levels = 4

deg = 2
degdif = 1

######################################################
# from bhdata import BHCurves
# fun_dw  = BHCurves(4)
# fun_w   = fun_dw.Integrate()
# fun_ddw = fun_dw.Differentiate()
######################################################
from bhdata import Brauer, BrauerCut
fun_w, fun_dw, fun_ddw  = BrauerCut()
######################################################

mu0 = 1.256636e-6
nu0 = 1/mu0

linear = "coil|ambient|default"
nonlinear = "r_steel|l_steel|mid_steel"

# linear = "r_steel|l_steel|mid_steel|coil|ambient|default"
# nonlinear = ""
######################################################



meshes = []; its = []; errorBs = []; errorHs = []



for i in range(levels):

    print("#####################################################")
    print(f"# Level {i}")
    print("#####################################################")

    with ngs.TaskManager():
        mesh = ngs.Mesh('whatever.vol')
    
    # Just for testing!
    # mesh.Refine()
    # mesh.Refine()

    for j in range(i):
        mesh.Refine()
    mesh.Curve(7)
    
    # with ngs.TaskManager(): J = mesh.MaterialCF({'coil_plus': (0,0,strom), 'coil_minus': (0,0,-strom), 'stator': (0,0,0)}, default = (0,0,0))

    #########################################################
    HcurlCoil = ngs.HCurl(mesh, order = deg, dirichlet = "coil_outer|coil_inner|coil_up|coil_down", definedon = 'coil', nograds = True)
    TotalCurrent = 3000
    coil_depth = 0.1

    n = ngs.specialcf.normal(mesh.dim)
    gfuDirichlet = ngs.GridFunction(HcurlCoil)
    g = ngs.CF([(0,0,TotalCurrent/coil_depth) if bc=="coil_inner" else (0,0,0) for bc in mesh.GetBoundaries()])
    gfuDirichlet.Set(g,ngs.BND)

    from ngsolve.krylovspace import CGSolver

    # Weak form
    T, Tstar = HcurlCoil.TnT()
    Kt = ngs.BilinearForm(ngs.curl(Tstar)*ngs.curl(T)*ngs.dx)
        
    # Assembly
    with ngs.TaskManager(): Kt.Assemble()

    r = - Kt.mat * gfuDirichlet.vec

    # Solving
    maxres = 1e-5
    maxit = 10000
                                
    Tsol = ngs.GridFunction(HcurlCoil)
    print("Solving...")

    with ngs.TaskManager():
        iterativeSolver = CGSolver(Kt.mat, freedofs = HcurlCoil.FreeDofs(), atol  = maxres,  maxiter = maxit)
        Tsol.vec.data = iterativeSolver * r + gfuDirichlet.vec

    # if len(iterativeSolver.residuals) == maxit: print("... Failure!")
    # else: print("... Success!")
    print(f"Number of iterations = {iterativeSolver.iterations}/{maxit} | Residual = {iterativeSolver.residuals[-1]}")

    # Draw(ngs.curl(Tsol), mesh, vectors = { "grid_size" : 150},clipping = {"x" : 0, "y" : 0, "z" : -1, "dist" : 0})
    with ngs.TaskManager():
        flux_bd = ngs.Integrate(ngs.curl(Tsol) * n, mesh, definedon = mesh.Boundaries("coil_cut_2"))
    print(flux_bd)

    Hcurl = ngs.HCurl(mesh, order = deg, nograds = True, dirichlet = 'ambient_face')
    u,v = Hcurl.TnT()

    K = ngs.BilinearForm(ngs.curl(u)*ngs.curl(v)*ngs.dx)
    c = ngs.Preconditioner(K, type = "local")

    with ngs.TaskManager(): K.Assemble()

    f = ngs.LinearForm(Hcurl)
    f +=  ngs.curl(Tsol)* ngs.curl(v) * ngs.dx

    with ngs.TaskManager(): f.Assemble()

    Hs = ngs.GridFunction(Hcurl)
    print("Solving...")

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
    print(flux_bd)

    J = ngs.curl(Hs)
    #########################################################

    HCurl_0 = ngs.HCurl(mesh, order = deg, nograds = True, dirichlet = 'outer_face|coil_plus_face|coil_minus_face|stator_face')
    L2_0 = ngs.L2(mesh, dim = 3, order = deg-1)
    A0 = ngs.GridFunction(HCurl_0)
    with ngs.TaskManager(): A0.Set(ngs.CF((0,0,0)))
    A0, it = do.solve(HCurl_0, A0, mesh, deg, J, fun_w, fun_dw, fun_ddw, linear, nonlinear)
    its.append(it)
    B0 = ngs.GridFunction(L2_0, nested = True)
    with ngs.TaskManager(): B0.Set(ngs.curl(A0), bonus_intorder = 10)

    normB0 = ngs.sqrt(B0*B0 + 1e-8)
    H0 = B0*mesh.MaterialCF({linear: nu0, nonlinear: fun_dw(normB0)/normB0}, default = nu0).Compile()

    
    HCurl_1 = ngs.HCurl(mesh, order = deg+degdif, nograds = True, dirichlet = 'outer_face|coil_plus_face|coil_minus_face|stator_face')
    L2_1 = ngs.L2(mesh, dim = 3, order = deg+degdif-1)
    A1 = ngs.GridFunction(HCurl_1)
    # with ngs.TaskManager(): A1.Set(A0, bonus_intorder = 10)
    with ngs.TaskManager(): A0.Set(ngs.CF((0,0,0)))
    A1, it = do.solve(HCurl_1, A1, mesh, deg+degdif, J, fun_w, fun_dw, fun_ddw, linear, nonlinear)
    B1 = ngs.GridFunction(L2_1, nested = True)
    with ngs.TaskManager(): B1.Set(ngs.curl(A1), bonus_intorder = 10)

    #########################################################


    normB1 = ngs.sqrt(B1*B1 + 1e-8)
    H1 = B1*mesh.MaterialCF({linear: nu0, nonlinear: fun_dw(normB1)/normB1}, default = nu0).Compile()


    with ngs.TaskManager(): 
        errorB = ngs.Integrate((B0-B1)**2, mesh, order = 2*(deg+degdif))**(1/2)/\
                 ngs.Integrate((B1)**2, mesh, order = 2*(deg+degdif))**(1/2)
        errorH = ngs.Integrate((H0-H1)**2, mesh, order = 2*(deg+degdif))**(1/2)/\
                 ngs.Integrate((H1)**2, mesh, order = 2*(deg+degdif))**(1/2)
    print(errorB)
    print(errorH)
    errorBs.append(errorB)
    errorHs.append(errorH)
    
    # mesh.Refine()
print(its)

import numpy as np
a = np.array(errorBs)
print(np.log2(a[:-1]/a[1:]))
print(errorBs)

b = np.array(errorHs)
print(np.log2(b[:-1]/b[1:]))
print(errorHs)