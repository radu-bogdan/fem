# from netgen import gui
from ngsolve import *
from ttictoc import tic, toc

# ngsglobals.msg_level = 0

# geometry setup
if 1:
    import geometries as geoms
    geo=geoms.Make2dGeometry2()
    Draw(geo)

# meshing
if 1:
    mm=0.001
    ngmesh=geo.GenerateMesh (maxh=1*mm)
    ngmesh.Save("coil.vol")
    mesh = Mesh(ngmesh)
    Draw(mesh)
    mesh.Refine()
    mesh.Refine()
    # mesh.Refine()
    mesh.Curve(5)

    print("using 2d mesh with ne=", mesh.ne, " elements and nv=", mesh.nv, " points")


# problem setup
if 1:  # block problem setup
    murc = 1400
    mur = CoefficientFunction([1, murc, 1, 1])
    mu0 = 1.257e-6
    mu = mu0*mur
    nu = 1/mu
    IN = 1*100*3
    A = 30*mm*mm 
    J = CoefficientFunction([0, 0, IN/A, -IN/A])
    print("current density j=", IN/A, "A/m2")

    rot = CoefficientFunction((0,1,-1,0),dims=(2,2))
    def curl2d(a):
        return rot*grad(a)

    p=3



# fem setup
if 1:
    print("using polynomial degree p =",p)

    fes = H1(mesh, order=p, dirichlet=[2])
    u,v = fes.TnT()
    gfu = GridFunction(fes)

    # cf = CoefficientFunction(spl_eng(sqrt(curl2d(gfu)*curl2d(gfu))/10))
    # cf = CoefficientFunction([0, gfu*gfu, 0, 0])

    a = BilinearForm(fes, symmetric=True)
    a += (nu*curl2d(u)*curl2d(v))*dx

    # c = Preconditioner(a, type="bddc")

    f = LinearForm(fes)
    f += J * v * dx

    with TaskManager():
        a.Assemble()
        f.Assemble()

    Draw(gfu, mesh, "a")
    Draw(curl2d(gfu), mesh, "b")
    Draw(nu*curl2d(gfu), mesh, "h")

# # solution
# if 1:
#     # tic()        
#     # solver = CGSolver(mat=a.mat, pre=c.mat, precision=1e-12, maxiter=200)
#     # gfu.vec.data = solver * f.vec
#     # print("iterative solver needed", toc(), "sec")

#     # tic()
#     # gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
#     # print("sparsecholesky needed", toc(), "sec")

#     tic()
#     gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="pardiso") * f.vec
#     print("pardiso needed", toc(), "sec")

# # post-processing and drawing
# if 1:
#     fesb = L2(mesh, order=p-1, dim=2) 
#     fesh = L2(mesh, order=p-1, dim=2) 

#     gfb = GridFunction(fesb)
#     gfh = GridFunction(fesh)

#     with TaskManager():
#         gfb.Set(curl2d(gfu))
#         gfh.Set(nu*gfb)
#         nrmH = Integrate( mu*gfh*gfh, mesh) 

#     print("   nrmh = ", nrmH)

#     Draw(gfu,mesh,'a')
#     Draw(curl2d(gfu),mesh,'b')
#     Draw(nu*curl2d(gfu),mesh,'h')


### now nonlinear problem

if 1:
    from bhdata import BHCurves
    ml=4
    print("using nonlinear material law", ml)
    fun_dw  = BHCurves(ml)
    fun_w   = fun_dw.Integrate()
    fun_ddw = fun_dw.Differentiate()
else:
    print("using linear material law with mur=", murc)
    def fun_w(x):
        return x*x/(2*mu0*murc)
    def fun_dw(x):
        return x/(mu0*murc)
    def fun_ddw(x):
        return 1/(mu0*murc)
    
# nonlinear solver loop
    
with TaskManager():
    maxit=30
    tol2=1e-8
    regb=1e-12

    print("starting nonlinear solver")
    b = curl2d(gfu)
    nrmb = sqrt(b*b+regb)
    tmp = (1/2*mu0)*b*b
    cf_w = CoefficientFunction([tmp, fun_w(nrmb), tmp, tmp])
    cf_J = J*gfu
    def funW():
        return Integrate(cf_w-cf_J, mesh)

    cf_nut = CoefficientFunction([1/mu0, fun_dw(nrmb)/nrmb, 1/mu0, 1/mu0]) 
    lf_dw = LinearForm(fes)
    lf_dw += (cf_nut*b*curl2d(v) - J*v) * dx
    def fundW():
        lf_dw.Assemble()
        return lf_dw


    Id = CoefficientFunction((1,0,0,1), dims=(2,2))
    bbt = CoefficientFunction((b[0]*b[0], b[0]*b[1], b[1]*b[0], b[1]*b[1]),dims=(2,2))
    fun1 = fun_dw(nrmb)/nrmb
    fun2 = (fun_ddw(nrmb) - fun_dw(nrmb)/nrmb)/(nrmb*nrmb)
    # cf_nu = CoefficientFunction([1/mu0, fun_ddw(nrmb),1/mu0, 1/mu0])
    # cf_nu = CoefficientFunction([1/mu0, fun1, 1/mu0, 1/mu0])
    cf_nu = CoefficientFunction([1/mu0*Id, fun1*Id+fun2*bbt, 1/mu0*Id, 1/mu0*Id])
    bf_da = BilinearForm(fes, symmetric=True)
    bf_da += (cf_nu*curl2d(u)*curl2d(v))*dx
    def funddW():
        bf_da.Assemble()
        return bf_da

    # solve again with newton
    Draw(b, mesh, "b")
    Draw(cf_nut * b, mesh, "h")

    gfu.Set(0)
    # Redraw()

    du = GridFunction(fes)
    uo = GridFunction(fes)
    wo = 1e12
    tic()
    for it in range(1,maxit+1):
        # with TaskManager():
        w = funW()
        dw = fundW()
        da = funddW()
        du.vec.data = da.mat.Inverse(fes.FreeDofs(), inverse="pardiso") * dw.vec 
        nrm = InnerProduct(du.vec,dw.vec)
        if it==1:
            nrm0=nrm

        print("it", it, ": w=",w, ", relres=", nrm/nrm0)

        if nrm/nrm0<tol2:
            print("converged to desired tolerance")
            break
        elif abs(wo-w)<tol2*1e-2:
            print("stopped early due to stagnation")
            break
        else:
            # linesearch
            uo.vec.data = gfu.vec.data
            wo = w
            alpha = 1
            for init in range(1,21):
                gfu.vec.data -= alpha*du.vec.data
                wn = funW()
                if wn < w - alpha*0.1*nrm:
                    print("alpha=",alpha)
                    break
                else:
                    alpha=alpha/2
                    gfu.vec.data = uo.vec.data

        Redraw()
    print("nonlinear solve required", toc(), "sec")


## post processing
if 1:
    fesb = L2(mesh, order=p-1, dim=2) 
    fesh = L2(mesh, order=p-1, dim=2) 

    gfb = GridFunction(fesb)
    gfh = GridFunction(fesh)

    with TaskManager():
        gfb.Set(curl2d(gfu))
        gfh.Set(cf_nut*curl2d(gfu))
        
# end of story

