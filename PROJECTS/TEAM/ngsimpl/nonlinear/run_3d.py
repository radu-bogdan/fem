import ngsolve as ngs
from do import solve
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt
from copy import deepcopy

levels = 3
deg = 2
degdif = 1


######################################################
# from bhdata import BHCurves
# fun_dw  = BHCurves(4)
# fun_w   = fun_dw.Integrate()
# fun_ddw = fun_dw.Differentiate()
######################################################
from bhdata import Brauer, BrauerCut
fun_w, fun_dw, fun_ddw  = Brauer()
######################################################

mu0 = 1.256636e-6
nu0 = 1/mu0

linear = "coil_plus|coil_minus"
nonlinear = "stator"

strom = 1e5
amps = 0.025**2*3.15*strom
print(f'Applying a current of {amps:.2f} Amps')
######################################################

from createGeom import makeGeo
geoOCC = makeGeo()




meshes = []; its = []; errors = []

for i in range(levels):

    print("#####################################################")
    print(f"# Level {i}")
    print("#####################################################")

    ngmesh = geoOCC.GenerateMesh()
    mesh = ngs.Mesh(ngmesh)

    for j in range(i):
        mesh.Refine()
    mesh.Curve(3)
    
    with ngs.TaskManager(): J = mesh.MaterialCF({'coil_plus': (0,0,strom), 'coil_minus': (0,0,-strom), 'stator': (0,0,0)}, default = (0,0,0))
    
    HCurl_0 = ngs.HCurl(mesh, order = deg, nograds = True, dirichlet = 'outer_face|coil_plus_face|coil_minus_face|stator_face')
    L2_0 = ngs.L2(mesh, dim = 3, order = deg-1)
    A0 = ngs.GridFunction(HCurl_0)
    A0, it = solve(HCurl_0, A0, mesh, deg, J, fun_w, fun_dw, fun_ddw, linear, nonlinear)
    # its.append(it)
    B0 = ngs.GridFunction(L2_0, nested = True)
    with ngs.TaskManager(): B0.Set(ngs.curl(A0), bonus_intorder = 10)

    
    HCurl_1 = ngs.HCurl(mesh, order = deg+degdif, nograds = True, dirichlet = 'outer_face|coil_plus_face|coil_minus_face|stator_face')
    L2_1 = ngs.L2(mesh, dim = 3, order = deg+degdif-1)
    A1 = ngs.GridFunction(HCurl_1)
    A1, it = solve(HCurl_1, A1, mesh, deg+degdif, J, fun_w, fun_dw, fun_ddw, linear, nonlinear)
    its.append(it)
    B1 = ngs.GridFunction(L2_1, nested = True)
    with ngs.TaskManager(): B1.Set(ngs.curl(A1), bonus_intorder = 10)

    with ngs.TaskManager(): error = ngs.Integrate((B0-B1)**2, mesh, order = 3*(deg+degdif))**(1/2)/\
                                    ngs.Integrate((B1)**2, mesh, order = 3*(deg+degdif))**(1/2)
    print(error)
    errors.append(error)
    
print(its)

import numpy as np
a = np.array(errors)
print(np.log2(a[:-1]/a[1:]))