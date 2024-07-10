import ngsolve as ngs
from do import solve, solve_2d
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt
from copy import deepcopy
import netgen.occ as occ

levels = 5
deg = 5
degdif = 1

######################################################
# from bhdata import BHCurves
# fun_dw  = BHCurves(3)
# fun_w   = fun_dw.Integrate()
# fun_ddw = fun_dw.Differentiate()
######################################################
from bhdata import Brauer, BrauerCut
fun_w, fun_dw, fun_ddw  = BrauerCut()
######################################################

mu0 = 1.256636e-6
nu0 = 1/mu0

linear = "coil_plus|coil_minus"
nonlinear = "stator"

# linear = "stator|coil_plus|coil_minus"
# nonlinear = ""

strom = 1e5
amps = 0.025**2*3.15*strom
print(f'Applying a current of {amps:.2f} Amps')
######################################################

r_coil = 0.025
y_coil = 0.05
r_outer = 0.1

c1 = occ.WorkPlane().Circle(r_outer).Face()
c2 = occ.WorkPlane().MoveTo(0,y_coil).Circle(r_coil).Face()
c3 = occ.WorkPlane().MoveTo(0,-y_coil).Circle(r_coil).Face()

full = occ.Glue([c1,c2,c3])

full.faces[0].name = 'stator'
full.faces[1].name = 'coil_plus'
full.faces[2].name = 'coil_minus'

full.edges[0].name = 'outer'

geoOCC = occ.OCCGeometry(full, dim = 2)
######################################################




meshes = []; its = []; errors = []

for i in range(levels):

    print(" ")
    print("#####################################################")
    print(f"# Level {i}")
    print("#####################################################")

    with ngs.TaskManager():
        ngmesh = geoOCC.GenerateMesh()
        mesh = ngs.Mesh(ngmesh)

        for j in range(i):
            mesh.Refine()
        mesh.Curve(7)

    with ngs.TaskManager(): J = mesh.MaterialCF({'coil_plus': strom, 'coil_minus': -strom, 'stator': 0}, default = 0)
    H1_0 = ngs.H1(mesh, order = deg, dirichlet = "outer")
    L2_0 = ngs.L2(mesh, dim = 2, order = deg-1)
    A0 = ngs.GridFunction(H1_0)
    with ngs.TaskManager(): A0.Set(ngs.CF((0)))
    A0, it = solve_2d(H1_0, A0, mesh, deg, J, fun_w, fun_dw, fun_ddw, linear, nonlinear)
    its.append(it)
    B0 = ngs.GridFunction(L2_0, nested = True)
    with ngs.TaskManager(): B0.Set(ngs.grad(A0), bonus_intorder = 10)


    H1_1 = ngs.H1(mesh, order = deg+degdif, dirichlet = "outer")
    L2_1 = ngs.L2(mesh, dim = 2, order = deg+degdif-1)
    A1 = ngs.GridFunction(H1_1)
    # with ngs.TaskManager(): A1.Set(A0, bonus_intorder = 10)
    with ngs.TaskManager(): A1.Set(ngs.CF((0)))
    A1, it = solve_2d(H1_1, A1, mesh, deg+degdif, J, fun_w, fun_dw, fun_ddw, linear, nonlinear)
    B1 = ngs.GridFunction(L2_1, nested = True)
    with ngs.TaskManager(): B1.Set(ngs.grad(A1), bonus_intorder = 10)

    with ngs.TaskManager(): 
        error = ngs.Integrate((B0-B1)**2, mesh, order = 3*(deg+degdif))**(1/2)/\
                ngs.Integrate((B1)**2, mesh, order = 3*(deg+degdif))**(1/2)
    print(error)
    errors.append(error)
    
print(its)

import numpy as np
a = np.array(errors)
print(np.log2(a[:-1]/a[1:]))
print(errors)


