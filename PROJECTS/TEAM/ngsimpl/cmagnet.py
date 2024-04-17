from netgen import gui
# from netgen.csg import *
import netgen.occ as occ
from ngsolve import *
from netgen.meshing import MeshingParameters


def MakeGeometry():
    hh=0.15; h1=0.25; h2=0.01; 
    
    box = occ.Box(occ.Pnt(-1,-1,-1), occ.Pnt(2,1,2))
    
    core = occ.Box(occ.Pnt(0,-0.05,0),occ.Pnt(0.8,0.05,1))- \
           occ.Box(occ.Pnt(0.1,-1,0.1),occ.Pnt(0.7,1,0.9))- \
           occ.Box(occ.Pnt(0.5,-1,0.4),occ.Pnt(1,1,0.6))
    
    core.mat("core")
    core.maxh = hh
    
    coil = (occ.Cylinder(occ.Pnt(0.05,0,0), occ.Z, r=0.3 ,h=1) - \
            occ.Cylinder(occ.Pnt(0.05,0,0), occ.Z, r=0.15,h=1)) * \
            occ.Box(occ.Pnt(-1,-1,0.3),occ.Pnt(1,1,0.7))
            
    coil.mat("coil")
    coil.maxh = hh
    
    for face in coil.faces: face.name = 'coil_faces'
    for face in core.faces: face.name = 'core_faces'
    for face in box.faces: face.name = 'box_faces'

    # for edge in core.edges: edge.maxh = h2
    # for edge in coil.edges: edge.maxh = h1
    for vertices in core.vertices: vertices.maxh = h2
        
    full = occ.Glue([box,core,coil])
    geoOCC = occ.OCCGeometry(full)
    # geoOCC.Draw()    
    # geoOCCmesh = geoOCC.GenerateMesh()    
    # ngsolveMesh = ng.Mesh(geoOCCmesh)
    return geoOCC

# geo, full = MakeGeometry2()
# geoOCCmesh = geo.GenerateMesh()
# ngsolveMesh = ng.Mesh(geo)
# DrawGeo(full)
# DrawGeo(full, clipping={"z": -1, "dist":1})
# DrawGeo(geoOCCmesh, clipping={"z": -1, "dist":1})

geo = MakeGeometry()
geo.Draw()

ngmesh = geo.GenerateMesh()
ngmesh.Save("coil.vol")
mesh = Mesh(ngmesh)
Draw(mesh)
 
# mesh.Refine()
# mesh.Refine()

# curve elements for geometry approximation
mesh.Curve(5)
ngsglobals.msg_level = 5

# polynomial order
p = 3


########################################################
## Problem parameters
########################################################

mur = mesh.MaterialCF({ "core" : 10000 }, default=1)
mu0 = 1.257e-6
mu = mu0*mur
nu = 1/mu
I = 1.5e7

# J = CoefficientFunction( (I*y,I*(0.05-x),0)) 
J = mesh.MaterialCF({ "coil" : (I*y,I*(0.05-x),0) }, default=(0,0,0)) 


#################################################################
## reference solution: Vector potential
#################################################################

fes = HCurl(mesh, order=p, dirichlet="outer", nograds = True)

# u and v refer to trial and test-functions in the definition of forms below
u,v = fes.TnT()
#
a = BilinearForm(fes, symmetric=True)
a += nu*curl(u)*curl(v)*dx + 1e-6*nu*u*v*dx

c = Preconditioner(a, type="bddc")
# c = Preconditioner(a, type="local", blocktype=4)

f = LinearForm(fes)
f += J * v * dx("coil")

gfu = GridFunction(fes)

with TaskManager():
    a.Assemble()
    f.Assemble()
    solver = CGSolver(mat=a.mat, pre=c.mat, precision=1e-10, maxiter=2000)
    gfu.vec.data = solver * f.vec


Draw (curl(gfu), mesh, "B")
# Draw (gfu.Deriv(), mesh, "B")
# Draw (nu*gfu.Deriv(), mesh, "H")

fesb = L2(mesh, order=p, dim=3)
fesh = L2(mesh, order=p, dim=3)

gfb = GridFunction(fesb)
gfh = GridFunction(fesh)

with TaskManager():
    gfb.Set(curl(gfu))
    gfh.Set(nu*curl(gfu))

Draw (gfb, mesh, "B")
Draw (gfh, mesh, "H")


##########################################################
## Reduced Scalar Potential Approach
##########################################################

eps0=1e-6

## precompute source field Hs : curl(Hs)=J
hs, dhs = fes.TnT()

aS = BilinearForm(fes, symmetric=True)
aS += (eps0*hs*dhs + curl(hs)*curl(dhs))*dx

cS = Preconditioner(aS, type="bddc")
# cS = Preconditioner(aS, type="local", blocktype=4)

fS = LinearForm(fes)
fS += (J*curl(dhs))*dx("coil")

gfHs = GridFunction(fes)

with TaskManager():
    aS.Assemble()
    fS.Assemble()
    solverS = CGSolver(mat=aS.mat, pre=cS.mat, precision=1e-12, maxiter=2000)
    gfHs.vec.data = solverS * fS.vec

Draw (gfHs, mesh, "Hs")
Draw (curl(gfHs), mesh, "Js")


## now start reduced scalar potential approach
fesR = H1(mesh, order=p+1)
psi, dpsi = fesR.TnT()

aR = BilinearForm(fesR, symmetric=True)
aR += (mu*grad(psi)*grad(dpsi)+eps0*mu*psi*dpsi)*dx

fR = LinearForm(fesR)
fR += (mu*gfHs*grad(dpsi))*dx

gfR = GridFunction(fesR)

cR = Preconditioner(aR, type="bddc")
# cR = Preconditioner(aR, type="local", blocktype=2)
with TaskManager():
    aR.Assemble()
    fR.Assemble()
    solverR = CGSolver(mat=aR.mat, pre=cR.mat, precision=1e-10, maxiter=2000)
    gfR.vec.data = solverR * fR.vec

# aR.Assemble()
# fR.Assemble()
# gfR.vec.data = aR.mat.Inverse(fesR.FreeDofs(), inverse="pardiso") * (fR.vec)

gfhR = GridFunction(fesh)
gfbR = GridFunction(fesb)

with TaskManager():
    gfhR.Set(gfHs-grad(gfR))
    gfbR.Set(mu*gfhR)

Draw (gfhR, mesh, 'HR')
Draw (gfbR, mesh, "BR")

##########################################################
## Regularized H-field Approximation
##########################################################

# order
eps0=1e-7
eps=nu*eps0

fesH = HCurl(mesh, order=p)

h,dh = fesH.TnT()

aH = BilinearForm(fesH, symmetric="True")
aH += (mu*h*dh + (1/eps)*curl(h)*curl(dh))*dx

cH = Preconditioner(aH, type="bddc")
# cH = Preconditioner(aH, type="local", blocktype=2)

fH = LinearForm(fesH)
fH += ((1/eps)*J*curl(dh))*dx("coil")

gfH = GridFunction(fesH)

with TaskManager():
    aH.Assemble()
    fH.Assemble()
    solverH = CGSolver(mat=aH.mat, pre=cH.mat, precision=1e-10, maxiter=2000)
    gfH.vec.data = solverH * fH.vec

Draw (gfH, mesh, "HH")

gfB = GridFunction(fesb)
gfJ = GridFunction(fesb)

with TaskManager():
    gfB.Set(mu*gfH)
    gfJ.Set(curl(gfH))

Draw (gfB, mesh, "BH")
Draw (gfJ, mesh, "JH")


with TaskManager():
    nrm = Integrate( mu*gfh*gfh, mesh)
    errHH = Integrate ( mu*(gfh-gfH)*(gfh-gfH), mesh)
    errHR = Integrate ( mu*(gfh-gfhR)*(gfh-gfhR), mesh)
    errHRH = Integrate ( mu*(gfhR-gfH)*(gfhR-gfH), mesh)

print ("rel-error-HH=", sqrt (errHH / nrm ))
print ("rel-error-HR=", sqrt ( errHR / nrm))
print ("rel-error-HRH=", sqrt ( errHRH / nrm ))
