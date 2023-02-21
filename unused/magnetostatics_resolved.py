import netgen.gui
# ngsolve stuff
from ngsolve import *
# basic geometry features (for the background mesh)
from netgen.csg import *
# visualization stuff
from ngsolve.internal import *
from netgen.csg import *
import ngsolve.internal as ngsint 
from ngsolve.solvers import *
from netgen.geom2d import SplineGeometry
import scipy.interpolate as intp
import numpy as np

resolved = False

geo=SplineGeometry()

geo.AddRectangle((0,0),(12,11),bc="diri", leftdomain=4,rightdomain=0)
pnts = [(2,4),(4,4),(4,7),(4.5,7),(7.5,7),(8,7),(8,4),(10,4),(10,9),(7.5,9),(4.5,9),(2,9)]
a,b,c,d,e,f,g,h,k,l,m,n=pind = [ geo.AppendPoint(*pnt) for pnt in pnts ]
pnts=[(2,3),(2,2),(10,2),(10,3),(8,3),(4,3),(4.5,6.5),(7.5,6.5),(7.5,9.5),(4.5,9.5)]
o,p,q,r,s,t,u,v,w,x=pind = [ geo.AppendPoint(*pnt) for pnt in pnts ]

geo.Append(['line',a,b],leftdomain=1,rightdomain=3)
geo.Append(['line',b,c],leftdomain=1,rightdomain=4)
geo.Append(['line',c,d],leftdomain=1,rightdomain=4)
geo.Append(['line',d,e],leftdomain=1,rightdomain=6)
geo.Append(['line',e,f],leftdomain=1,rightdomain=4)
geo.Append(['line',f,g],leftdomain=1,rightdomain=4)
geo.Append(['line',g,h],leftdomain=1,rightdomain=3)
geo.Append(['line',h,k],leftdomain=1,rightdomain=4)
geo.Append(['line',k,l],leftdomain=1,rightdomain=4)
geo.Append(['line',l,m],leftdomain=1,rightdomain=5)
geo.Append(['line',m,n],leftdomain=1,rightdomain=4)
geo.Append(['line',n,a],leftdomain=1,rightdomain=4)
geo.Append(['line',a,o],leftdomain=3,rightdomain=4)
geo.Append(['line',o,t],leftdomain=3,rightdomain=2,bc="Omega2_top")
geo.Append(['line',t,b],leftdomain=3,rightdomain=4)
geo.Append(['line',g,s],leftdomain=3,rightdomain=4)
geo.Append(['line',s,r],leftdomain=3,rightdomain=2,bc="Omega2_top")
geo.Append(['line',r,h],leftdomain=3,rightdomain=4)
geo.Append(['line',o,p],leftdomain=2,rightdomain=4,bc="Omega2_left")
geo.Append(['line',p,q],leftdomain=2,rightdomain=4,bc="Omega2_bottom")
geo.Append(['line',q,r],leftdomain=2,rightdomain=4,bc="Omega2_right")
geo.Append(['line',s,t],leftdomain=2,rightdomain=4,bc="Omega2_top")
geo.Append(['line',l,w],leftdomain=5,rightdomain=4)
geo.Append(['line',w,x],leftdomain=5,rightdomain=4)
geo.Append(['line',x,m],leftdomain=5,rightdomain=4)
geo.Append(['line',d,u],leftdomain=6,rightdomain=4)
geo.Append(['line',u,v],leftdomain=6,rightdomain=4)
geo.Append(['line',v,e],leftdomain=6,rightdomain=4)

if resolved:
    pnts = [(0,4),(2,4),(4,4),(8,4),(10,4),(12,4)]
    o0,p0,q0,r0,s0,t0  = pind = [ geo.AppendPoint(*pnt) for pnt in pnts ]
    geo.Append(['line',o0,p0],leftdomain=4,rightdomain=4)
    geo.Append(['line',q0,r0],leftdomain=4,rightdomain=4)
    geo.Append(['line',s0,t0],leftdomain=4,rightdomain=4)

    pnts=[(0,3),(0,2),(12,2),(12,3)]
    o1,p1,q1,r1 = pind = [ geo.AppendPoint(*pnt) for pnt in pnts ]
    geo.Append(['line',o,o1],leftdomain=4,rightdomain=4)
    geo.Append(['line',p,p1],leftdomain=4,rightdomain=4)
    geo.Append(['line',q,q1],leftdomain=4,rightdomain=4)
    geo.Append(['line',r,r1],leftdomain=4,rightdomain=4)

    pnts=[(0,1),(12,1)]
    o2,p2 = pind = [ geo.AppendPoint(*pnt) for pnt in pnts ]
    geo.Append(['line',o2,p2],leftdomain=4,rightdomain=4)

geo.SetMaterial(1,"Omega1")
geo.SetMaterial(2,"Omega2")
geo.SetMaterial(3,"Omega3")
geo.SetMaterial(4,"Omega4")
geo.SetMaterial(5,"Omega5")
geo.SetMaterial(6,"Omega6")

mesh= Mesh (geo.GenerateMesh(maxh=0.1,quad_dominated=False))

nu_iron=1e-3
nu0=1
p=0
dp=1e-4

def deformation(p):
    return CoefficientFunction((0,IfPos(y-1,1,0)*IfPos(y-2,0,1)*p*(y-1)+IfPos(y-2,1,0)*IfPos(y-3,0,1)*p+IfPos(y-3,1,0)*IfPos(y-4,0,1)*p*(4-y)))

def moveNGmesh(displ,mesh):
    for p in mesh.ngmesh.Points():
        mip=mesh(p[0],p[1])
        v=displ(mip)
        p[1] +=v[1]
    mesh.ngmesh.Update()
    
nucf=CoefficientFunction([nu_iron,nu_iron,nu0,nu0,nu0,nu0])
I=CoefficientFunction([0,0,0,0,-1,1])
RotM=CoefficientFunction((0,1,-1,0),dims=(2,2))

V=H1(mesh,order=1,dirichlet="diri")
gfu=GridFunction(V)
u,v=V.TnT()
a=BilinearForm(V,symmetric=True)
a+=nucf*grad(u)*grad(v)*dx
f=LinearForm(V)
f+=I*v*dx
a.Assemble()
f.Assemble()
gfu.vec.data=a.mat.Inverse(freedofs=V.FreeDofs())*f.vec

Draw(gfu,mesh,"gfu")
Draw(RotM*grad(gfu),mesh,"B")
Draw(nucf*RotM*grad(gfu),mesh,"H")
Draw(nucf/2*(grad(gfu)[0]*grad(gfu)[0]+grad(gfu)[1]*grad(gfu)[1])-I*gfu,mesh,"e")

#input("Part A and B. Press Enter to continue")

E0=Integrate(cf=nucf/2*grad(gfu)*grad(gfu)-I*gfu,mesh=mesh,order=1)
print("Energy",E0)
#p=float(input("Input p: "))
p = 0.0

moveNGmesh(deformation(p),mesh)
a.Assemble()
f.Assemble()
gfu.vec.data=a.mat.Inverse(freedofs=V.FreeDofs())*f.vec
Redraw()

Ep=Integrate(cf=nucf/2*grad(gfu)*grad(gfu)-I*gfu,mesh=mesh,order=1)

print("Energy(p)",Ep)
#dp=float(input("Input dp: "))
dp = 0.000001

moveNGmesh(deformation(dp),mesh)
a.Assemble()
f.Assemble()
gfu.vec.data=a.mat.Inverse(freedofs=V.FreeDofs())*f.vec
Redraw()

Edp=Integrate(cf=nucf/2*grad(gfu)*grad(gfu)-I*gfu,mesh=mesh,order=1)

print("Energy(p+dp)",Edp)

Fp=-(Edp-Ep)/dp

print("Force(p)",Fp)
#input("Press Enter to continue")

moveNGmesh(deformation(-2*dp),mesh)
a.Assemble()
f.Assemble()
gfu.vec.data=a.mat.Inverse(freedofs=V.FreeDofs())*f.vec
Redraw()

Emdp=Integrate(cf=nucf/2*grad(gfu)*grad(gfu)-I*gfu,mesh=mesh,order=1)

print("Energy(p-dp)",Emdp)

Fcp=-(Edp-Emdp)/2/dp

print("Force(p) (central)",Fcp)
#input("Press Enter to continue")

moveNGmesh(deformation(dp),mesh)

V2=H1(mesh,order=1,dirichlet="diri", dim=2)
teta = GridFunction(V2)
teta.Set(deformation(1))

#A = GridFunction(V)
#A.Set(CoefficientFunction(grad(gfu)*grad(gfu)))
A = grad(gfu)*grad(gfu)
B1 = (grad(teta)[0]*grad(gfu)[0] + grad(teta)[1]*grad(gfu)[1])*grad(gfu)[0]
B2 = (grad(teta)[2]*grad(gfu)[0] + grad(teta)[3]*grad(gfu)[1])*grad(gfu)[1]
B = B1 - B2

j1 = Integrate( nucf/2 * A * (grad(teta)[0]+grad(teta)[3]) * dx, mesh)
j2 = Integrate( nucf * B * dx, mesh)
#print("j1, j2", j1, j2)
Fcp_shape = -( j1 + j2 )
print("Force(p) (shape derivative vol) ",Fcp_shape)

mod2 = GridFunction(V)
mod2.Set(CoefficientFunction(grad(gfu)*grad(gfu)))

gradx_2 = grad(gfu)[0]*grad(gfu)[0]
grady_2 = grad(gfu)[1]*grad(gfu)[1]

j1 = Integrate( -1*( (nu_iron-nu0)/2*gradx_2 - (nu_iron-nu0)/2*grady_2 ) * ds("Omega2_bottom"), mesh)
j2 = Integrate( ( (nu_iron-nu0)/2*gradx_2 - (nu_iron-nu0)/2*grady_2 ) * ds("Omega2_top"), mesh)
j3 = Integrate( grad(gfu)[0] * ds("Omega2_top"), mesh)
j4 = Integrate( grad(gfu)[1] * ds("Omega2_top"), mesh)

Fcp_shape_1d = - (j1 + j2)
print("Force(p) (shape derivative bnd) ",Fcp_shape_1d)
#print("j1, j2 ", j1, j2)
#print("j3, j4 ", j3, j4)

Draw(gfu,mesh,"gfu")
Draw(grad(gfu),mesh,"grad_gfu")
Draw(teta[0],mesh,"teta1")
Draw(teta[1],mesh,"teta2")
Draw(teta,mesh,"teta")
Draw(grad(teta),mesh,"grad_teta")
