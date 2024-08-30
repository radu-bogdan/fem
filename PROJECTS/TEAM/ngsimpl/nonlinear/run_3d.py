import ngsolve as ngs
from do import solve
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt
from copy import deepcopy
import pyngcore as ngcore
ngcore.SetNumThreads(48)

ngs.ngsglobals.msg_level = 0


# import netgen.meshing
# netgen.meshing.Mesh.EnableTableClass("edges", True)
# netgen.meshing.Mesh.EnableTableClass("faces", True)

levels = 5
deg = 3
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

linear = "coil_plus|coil_minus"
nonlinear = "stator"

strom = 1e5
amps = 0.025**2*3.1415926*strom
print(f'Applying a current of {amps:.2f} Amps')
######################################################

from createGeom import makeGeo
geoOCC = makeGeo()



meshes = []; its = []; errorBs = []; errorHs = []



for i in range(levels):

    print("#####################################################")
    print(f"# Level {i}")
    print("#####################################################")

    with ngs.TaskManager():
        ngmesh = geoOCC.GenerateMesh(maxh = 1)
        mesh = ngs.Mesh(ngmesh)

    for j in range(i):
        mesh.Refine()
    mesh.Curve(7)
    
    with ngs.TaskManager(): J = mesh.MaterialCF({'coil_plus': (0,0,strom), 'coil_minus': (0,0,-strom), 'stator': (0,0,0)}, default = (0,0,0))
    HCurl_0 = ngs.HCurl(mesh, order = deg, nograds = True, dirichlet = 'outer_face|coil_plus_face|coil_minus_face|stator_face')
    L2_0 = ngs.L2(mesh, dim = 3, order = deg-1)
    A0 = ngs.GridFunction(HCurl_0)
    with ngs.TaskManager(): A0.Set(ngs.CF((0,0,0)))
    A0, it = solve(HCurl_0, A0, mesh, deg, J, fun_w, fun_dw, fun_ddw, linear, nonlinear)
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
    A1, it = solve(HCurl_1, A1, mesh, deg+degdif, J, fun_w, fun_dw, fun_ddw, linear, nonlinear)
    B1 = ngs.GridFunction(L2_1, nested = True)
    with ngs.TaskManager(): B1.Set(ngs.curl(A1), bonus_intorder = 10)


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