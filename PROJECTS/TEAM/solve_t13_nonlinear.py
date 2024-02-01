import sys
sys.path.insert(0,'../../') # adds parent directory
from scipy import sparse as sp

from solve_t13_strom import *
from nonlin_TEAM13 import *

MESH = pde.mesh3.netgen(geoOCCmesh)

##############################################################################
# B-H curves
##############################################################################

linear = '*coil,default'
nonlinear = 'r_steel,l_steel,mid_steel'
maxIter = 100

##############################################################################
# Tree/Cotree gauging
##############################################################################

R = pde.tools.tree_cotree_gauge(MESH,random_edges=True)

##############################################################################
# Assembly
##############################################################################

order = 1

fem_linear = pde.int.evaluate3(MESH, order = order, regions = linear).diagonal()
fem_nonlinear = pde.int.evaluate3(MESH, order = order, regions = nonlinear).diagonal()

D = pde.int.assemble3(MESH, order = order)

phix_Hcurl, phiy_Hcurl, phiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'M', order = order)
curlphix_Hcurl, curlphiy_Hcurl, curlphiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'K', order = order)

aJ = jx_L2 @ D @ phix_Hcurl.T +\
     jy_L2 @ D @ phiy_Hcurl.T +\
     jz_L2 @ D @ phiz_Hcurl.T
     
aJ = 1e7*aJ

def gss(A):
    curl_Ax = curlphix_Hcurl.T@A; curl_Ay = curlphiy_Hcurl.T@A; curl_Az = curlphiz_Hcurl.T@A
    
    Kxx = curlphix_Hcurl @ D @ sp.diags(fxx_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fxx_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear)@ curlphix_Hcurl.T
    Kyy = curlphiy_Hcurl @ D @ sp.diags(fyy_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fyy_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear)@ curlphiy_Hcurl.T
    Kzz = curlphiz_Hcurl @ D @ sp.diags(fzz_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fzz_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear)@ curlphiz_Hcurl.T
    
    Kxy = curlphiy_Hcurl @ D @ sp.diags(fxy_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fxy_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear)@ curlphix_Hcurl.T
    Kxz = curlphiz_Hcurl @ D @ sp.diags(fxz_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fxz_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear)@ curlphix_Hcurl.T
    
    Kyx = curlphix_Hcurl @ D @ sp.diags(fyx_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fyx_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear)@ curlphiy_Hcurl.T
    Kyz = curlphiz_Hcurl @ D @ sp.diags(fyz_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fyz_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear)@ curlphiy_Hcurl.T
    
    Kzx = curlphix_Hcurl @ D @ sp.diags(fzx_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fzx_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear)@ curlphiz_Hcurl.T
    Kzy = curlphiy_Hcurl @ D @ sp.diags(fzy_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fzy_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear)@ curlphiz_Hcurl.T
    
    return Kxx + Kyy + Kzz + Kxy + Kxz + Kyx + Kyz + Kzx + Kzy
    
def gs(A):
    curl_Ax = curlphix_Hcurl.T@A; curl_Ay = curlphiy_Hcurl.T@A; curl_Az = curlphiz_Hcurl.T@A
    return curlphix_Hcurl @ D @ (fx_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fx_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear) +\
           curlphiy_Hcurl @ D @ (fy_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fy_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear) +\
           curlphiz_Hcurl @ D @ (fz_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + fz_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear) - aJ
           
def J(A):
    curl_Ax = curlphix_Hcurl.T@A; curl_Ay = curlphiy_Hcurl.T@A; curl_Az = curlphiz_Hcurl.T@A
    return np.ones(D.size)@ D @(f_linear(curl_Ax,curl_Ay,curl_Az)*fem_linear + f_nonlinear(curl_Ax,curl_Ay,curl_Az)*fem_nonlinear) -aJ@A





A = np.zeros(curlphix_Hcurl.shape[0])
mu = 0.0001
eps_newton = 1e-5
factor_residual = 1/2

tm = time.monotonic()
for i in range(maxIter):
    
    gssu = R.T @ gss(A) @ R
    gsu = R.T @ gs(A)
    
    tm = time.monotonic()
    # wS = chol(gssu).solve_A(-gsu)
    wS = sp.linalg.spsolve(gssu, -gsu)
    print('Solving took ', time.monotonic()-tm)
    
    w = R@wS
    
    alpha = 1
    
    # ResidualLineSearch
    # for k in range(1000):
    #     if np.linalg.norm(gs(A+alpha*w),np.inf) <= np.linalg.norm(gs(A),np.inf): break
    #     else: alpha = alpha*factor_residual
    
    # AmijoBacktracking
    float_eps = 1e-12; #float_eps = np.finfo(float).eps
    for kk in range(1000):
        if J(A+alpha*w)-J(A) <= alpha*mu*(gsu@wS) + np.abs(J(A))*float_eps: break
        else: alpha = alpha*factor_residual
    
    
    A_old_i = A
    A = A + alpha*w
    
    print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(A)+"|| ||grad||: %2e" %np.linalg.norm(R.T @ gs(A),np.inf)+"||alpha: %2e" % (alpha))
    
    
    # if ( np.linalg.norm(R.T @ gs(A),np.inf) < eps_newton):
    #     break
    if (np.abs(J(A)-J(A_old_i)) < 1e-5):
        break
    
elapsed = time.monotonic()-tm
print('Solving took ', elapsed, 'seconds')









# K_Hcurl = curlphix_Hcurl @ D @ curlphix_Hcurl.T +\
#           curlphiy_Hcurl @ D @ curlphiy_Hcurl.T +\
#           curlphiz_Hcurl @ D @ curlphiz_Hcurl.T

# KR = R.T@K_Hcurl@R

# r = jx_L2 @ D @ phix_Hcurl.T +\
#     jy_L2 @ D @ phiy_Hcurl.T +\
#     jz_L2 @ D @ phiz_Hcurl.T


# cholKR = chol(KR)
# A = R @ cholKR.solve_A(R.T@r)

curlphix_Hcurl_P0, curlphiy_Hcurl_P0, curlphiz_Hcurl_P0 = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'K', order = 0)
Bx = curlphix_Hcurl_P0.T @ A
By = curlphiy_Hcurl_P0.T @ A
Bz = curlphiz_Hcurl_P0.T @ A

##############################################################################
# Storing to vtk
##############################################################################

import vtklib

grid = vtklib.createVTK(MESH)
vtklib.add_H1_Scalar(grid, potential_H1, 'potential_H1')
vtklib.add_L2_Vector(grid,jx_L2,jy_L2,jz_L2,'j_L2')
vtklib.add_L2_Vector(grid,Bx,By,Bz,'B')

vtklib.writeVTK(grid, 'das2.vtu')