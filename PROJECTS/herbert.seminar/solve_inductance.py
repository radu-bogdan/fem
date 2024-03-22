# import sys
# sys.path.insert(0,'../../') # adds parent directory
# import pde

import numpy as np
import ngsolve as ng
from geo_coil import *

mesh = ng.Mesh(geoOCCmesh)

import ngsolve as ng


order = 1
J = 8*66
s = 58


fes = ng.H1(mesh = mesh, order = order, dirichlet='outer')
u,v = fes.TnT()

sigma = mesh.MaterialCF({"coil_plus": s, "coil_minus": s}, default=1)

gfu = ng.GridFunction(fes)
gfu.vec.data = GradGrad.mat.Inverse(fes.FreeDofs()) * f.vec
Draw(gfu);

GradGrad = ng.BilinearForm(fes)
GradGrad += ng.grad(u)*ng.grad(v)*ng.dx
GradGrad.Assemble()

coil_plus  =  J * v * ng.dx(definedon = mesh.Materials("coil_plus"))
coil_minus = -J * v * ng.dx(definedon = mesh.Materials("coil_minus"))
f = ng.LinearForm(coil_plus + coil_minus).Assemble(); # fnp = f.vec.FV().NumPy()




# gfu = ng.GridFunction(fes)
# gfu.vec.data = GradGrad.mat.Inverse(fes.FreeDofs()) * f.vec
# Draw(gfu);

# gf2 = GridFunction(fes)
# gf2.Set(ng.CoefficientFunction(J), definedon = mesh.Boundaries("coil_plus"))
# Draw(gf2);


ir = ng.IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6] )
a =  ng.SymbolicBFI (u*v, intrule=ir)


# import 