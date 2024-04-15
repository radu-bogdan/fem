print('solve_t13_scal_pot_nl')

import sys
sys.path.insert(0,'../../') # adds parent directory
from scipy import sparse as sps

from nonlin_TEAM13_new import *
from solve_t13_mag_pot_lin import A
from solve_t13_strom import *

# @profile
# def do():

MESH = pde.mesh3.netgen(geoOCCmesh)

##############################################################################
# B-H curves
##############################################################################

order = 0

linear = '*coil,default'
nonlinear = 'r_steel,l_steel,mid_steel'
maxIter = 1000

curlphix_Hcurl, curlphiy_Hcurl, curlphiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'K', order = order)

Hjx = 1*curlphix_Hcurl.T @ A # 1 is "mu0" here
Hjy = 1*curlphiy_Hcurl.T @ A
Hjz = 1*curlphiz_Hcurl.T @ A

##############################################################################
# Assembly
##############################################################################

D = pde.int.assemble3(MESH, order = order)

# phi_H1  = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = 1)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = order)

fem_linear = pde.int.evaluate3(MESH, order = order, regions = linear).diagonal()
fem_nonlinear = pde.int.evaluate3(MESH, order = order, regions = nonlinear).diagonal()


# @profile
def gss(u):
    ux = dphix_H1.T@u; uy = dphiy_H1.T@u; uz = dphiz_H1.T@u
    
    Kxx = dphix_H1 @ D @ sps.diags(gxx_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gxx_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear)@ dphix_H1.T
    Kyy = dphiy_H1 @ D @ sps.diags(gyy_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gyy_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear)@ dphiy_H1.T
    Kzz = dphiz_H1 @ D @ sps.diags(gzz_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gzz_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear)@ dphiz_H1.T
    
    Kxy = dphiy_H1 @ D @ sps.diags(gxy_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gxy_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear)@ dphix_H1.T
    Kxz = dphiz_H1 @ D @ sps.diags(gxz_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gxz_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear)@ dphix_H1.T
    
    Kyx = Kxy.T # Kyx = dphix_H1 @ D @ sps.diags(gyx_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gyx_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear)@ dphiy_H1.T
    Kyz = dphiz_H1 @ D @ sps.diags(gyz_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gyz_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear)@ dphiy_H1.T
    
    Kzx = Kxz.T # Kzx = dphix_H1 @ D @ sps.diags(gzx_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gzx_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear)@ dphiz_H1.T
    Kzy = Kyz.T # Kzy = dphiy_H1 @ D @ sps.diags(gzy_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gzy_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear)@ dphiz_H1.T
    
    return Kxx + Kyy + Kzz + Kxy + Kxz + Kyx + Kyz + Kzx + Kzy 
    
def gs(u):
    ux = dphix_H1.T@u; uy = dphiy_H1.T@u; uz = dphiz_H1.T@u
    return dphix_H1 @ D @ (gx_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gx_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear) +\
           dphiy_H1 @ D @ (gy_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gy_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear) +\
           dphiz_H1 @ D @ (gz_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + gz_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear)

def J(u):
    ux = dphix_H1.T@u; uy = dphiy_H1.T@u; uz = dphiz_H1.T@u
    return np.ones(D.size)@ D @(g_linear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_linear + g_nonlinear(Hjx+ux,Hjy+uy,Hjz+uz)*fem_nonlinear)



R_out, RS = pde.h1.assembleR3(MESH, space = 'P1', faces = 'ambient_face')

u = np.zeros(MESH.np)
mu = 1e-2
# mu = 1/2
eps_newton = 1e-5
factor_residual = 1/2

for i in range(maxIter):
    tm2 = time.monotonic()
    
    tm = time.monotonic()
    gssu = RS @ gss(u) @ RS.T
    gsu = RS @ gs(u)
    asm = time.monotonic()-tm
    
    tm = time.monotonic()
    wS = chol(gssu).solve_A(-gsu)
    solt = time.monotonic()-tm
    
    # wS = pysolve(gssu,-gsu)
    # wS = sps.linalg.spsolve(gssu,-gsu)
    
    w = RS.T@wS
    
    alpha = 1
    
    # ResidualLineSearch
    # for k in range(1000):
    #     if np.linalg.norm(gs(u+alpha*w),np.inf) <= np.linalg.norm(gs(u),np.inf): break
    #     else: alpha = alpha*factor_residual
    
    # AmijoBacktracking
    float_eps = 1e-16; #float_eps = np.finfo(float).eps
    Ju = J(u)
    for kk in range(1000):
        if np.isnan(J(u+alpha*w)): print('nan action')
        if J(u+alpha*w)-Ju <= alpha*mu*(gsu@wS) + np.abs(Ju)*float_eps: break
        else: alpha = alpha*factor_residual
    
    u_old_i = u
    u = u + alpha*w
    
    residual = wS.T@gssu@wS
    residual2 = np.abs((RS@u)@gsu)
    
    print ("NEWTON : %2d " %(i+1)+"||obj: %.9e" %J(u)+"|| ||grad||: %.2e" %residual2 +"||alpha: %.2e" % (alpha) + "|| Step took : %.2f" %(time.monotonic()-tm2)+ "|| Assembly took : %.2f" %(asm)+ "|| Solution took : %.2f" %(solt))
    
    # if (residual2  < eps_newton):
    #     break
    
    # if (np.abs(J(u)-J(u_old_i)) < 1e-8*(np.abs(J(u))+np.abs(J(u_old_i))+1)):
    if np.abs(J(u)-J(u_old_i)) < 1e-8:
        break
    
elapsed = time.monotonic()-tm2
print('Solving took ', elapsed, 'seconds')

curlphix_Hcurl_P0, curlphiy_Hcurl_P0, curlphiz_Hcurl_P0 = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'K', order = 0)
dphix_H1_P0, dphiy_H1_P0, dphiz_H1_P0 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = 0)

ux = dphix_H1_P0.T@u; uy = dphiy_H1_P0.T@u; uz = dphiz_H1_P0.T@u
Bx_lin_P0 = curlphix_Hcurl_P0.T @ A
By_lin_P0 = curlphiy_Hcurl_P0.T @ A
Bz_lin_P0 = curlphiz_Hcurl_P0.T @ A

Hjx_P0 = Bx_lin_P0
Hjy_P0 = By_lin_P0
Hjz_P0 = Bz_lin_P0


Hx = Hjx_P0 + ux
Hy = Hjy_P0 + uy
Hz = Hjz_P0 + uz

fem_linear_P0 = pde.int.evaluate3(MESH, order = 0, regions = linear).diagonal()
fem_nonlinear_P0 = pde.int.evaluate3(MESH, order = 0, regions = nonlinear).diagonal()


Bx_new = gx_linear(Hx,Hy,Hz)*fem_linear_P0 + gx_nonlinear(Hx,Hy,Hz)*fem_nonlinear_P0
By_new = gy_linear(Hx,Hy,Hz)*fem_linear_P0 + gy_nonlinear(Hx,Hy,Hz)*fem_nonlinear_P0
Bz_new = gz_linear(Hx,Hy,Hz)*fem_linear_P0 + gz_nonlinear(Hx,Hy,Hz)*fem_nonlinear_P0

##############################################################################
# Storing to vtk
##############################################################################

grid = pde.tools.vtklib.createVTK(MESH)
pde.tools.vtklib.add_L2_Vector(grid,Bx_new,By_new,Bz_new,'B_new')
pde.tools.vtklib.add_L2_Vector(grid,Hx,Hy,Hz,'H_new')
pde.tools.vtklib.add_L2_Vector(grid,Bx_lin_P0,By_lin_P0,Bz_lin_P0,'B')
pde.tools.vtklib.writeVTK(grid, 'scalar_potential.vtu')
    
print(np.sqrt(Bx_new**2+By_new**2+Bz_new**2).max())
# do()