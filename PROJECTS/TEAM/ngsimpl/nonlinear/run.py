import ngsolve as ngs
from do import solve
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt
from copy import deepcopy

levels = 4
deg = 1

######################################################
from bhdata import BHCurves
fun_dw  = BHCurves(4)
fun_w   = fun_dw.Integrate()
fun_ddw = fun_dw.Differentiate()

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


meshes = []; Bs = []; Bps = []; its = [];

for i in range(levels):
    ngmesh = geoOCC.GenerateMesh()
    mesh = ngs.Mesh(ngmesh)

    for j in range(i):
        mesh.Refine()
    mesh.Curve(3)

    with ngs.TaskManager(): J = mesh.MaterialCF({'coil_plus': (0,0,strom), 'coil_minus': (0,0,-strom), 'stator': (0,0,0)}, default = (0,0,0))
    HCurl = ngs.HCurl(mesh, order = deg, nograds = True, dirichlet = 'outer_face|coil_plus_face|coil_minus_face|stator_face', autoupdate = True)
    V3L2 = ngs.L2(mesh, dim = 3, order = deg-1, autoupdate = True)
    A0 = ngs.GridFunction(HCurl)
    A0, it = solve(HCurl, A0, mesh,deg, J, fun_w, fun_dw, fun_ddw)
    its.append(it)

    B = ngs.GridFunction(V3L2, nested = True)
    with ngs.TaskManager(): B.Set(ngs.curl(A0))
    Bs.append(B)

    if i>0:
        V3L2 = ngs.L2(mesh, dim = 3, order = deg-1)
        Bp = ngs.GridFunction(V3L2)
        with ngs.TaskManager(): Bp.Set(Bs[i-1])
        Bps.append(Bp)
        with ngs.TaskManager(): error = ngs.Integrate((B-Bp)**2, mesh)*(1/2)
        print(error)

print(its)
