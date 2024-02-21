print('solve_t13_scal_pot_mix_nl')

from imports import *

from nonlin_TEAM13 import *
from solve_t13_strom import *
from solve_t13_mag_pot_lin import A

# @profile
# def do():
    
MESH = pde.mesh3.netgen(geoOCCmesh)
MESH.p = 1/1000*MESH.p

##############################################################################
# B-H curves
##############################################################################

order = 0

linear = '*coil,default'
nonlinear = 'r_steel,l_steel,mid_steel'
maxIter = 100

curlphix_Hcurl, curlphiy_Hcurl, curlphiz_Hcurl = pde.hcurl.assemble3(MESH, space = 'N0', matrix = 'K', order = order)

Hjx = 1*curlphix_Hcurl.T @ A # 1 is "mu0" here
Hjy = 1*curlphiy_Hcurl.T @ A
Hjz = 1*curlphiz_Hcurl.T @ A

##############################################################################
# Assembly
##############################################################################

D = pde.int.assemble3(MESH, order = order)

dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = order)
phi_L2 = pde.l2.assemble3(MESH, space = 'P0', matrix = 'M', order = order)

fem_linear = pde.int.evaluate3(MESH, order = order, regions = linear).diagonal()
fem_nonlinear = pde.int.evaluate3(MESH, order = order, regions = nonlinear).diagonal()

Cx = phi_L2 @ D @ dphix_H1.T
Cy = phi_L2 @ D @ dphiy_H1.T
Cz = phi_L2 @ D @ dphiz_H1.T


def gss(b):
    bx = b[:len(b)//3]; by = b[len(b)//3:2*len(b)//3]; bz = b[2*len(b)//3:]
    
    Kxx = phi_L2 @ D @ sp.diags(fxx_linear(bx,by,bz)*fem_linear + fxx_nonlinear(bx,by,bz)*fem_nonlinear)@ phi_L2.T
    Kyy = phi_L2 @ D @ sp.diags(fyy_linear(bx,by,bz)*fem_linear + fyy_nonlinear(bx,by,bz)*fem_nonlinear)@ phi_L2.T
    Kzz = phi_L2 @ D @ sp.diags(fzz_linear(bx,by,bz)*fem_linear + fzz_nonlinear(bx,by,bz)*fem_nonlinear)@ phi_L2.T
    
    Kxy = phi_L2 @ D @ sp.diags(fxy_linear(bx,by,bz)*fem_linear + fxy_nonlinear(bx,by,bz)*fem_nonlinear)@ phi_L2.T
    Kxz = phi_L2 @ D @ sp.diags(fxz_linear(bx,by,bz)*fem_linear + fxz_nonlinear(bx,by,bz)*fem_nonlinear)@ phi_L2.T
    
    Kyx = phi_L2 @ D @ sp.diags(fyx_linear(bx,by,bz)*fem_linear + fyx_nonlinear(bx,by,bz)*fem_nonlinear)@ phi_L2.T
    Kyz = phi_L2 @ D @ sp.diags(fyz_linear(bx,by,bz)*fem_linear + fyz_nonlinear(bx,by,bz)*fem_nonlinear)@ phi_L2.T
    
    Kzx = phi_L2 @ D @ sp.diags(fzx_linear(bx,by,bz)*fem_linear + fzx_nonlinear(bx,by,bz)*fem_nonlinear)@ phi_L2.T
    Kzy = phi_L2 @ D @ sp.diags(fzy_linear(bx,by,bz)*fem_linear + fzy_nonlinear(bx,by,bz)*fem_nonlinear)@ phi_L2.T
    
    
    R = bmat([[Kxx,Kxy,Kxz],
              [Kyx,Kyy,Kyz],
              [Kzx,Kzy,Kzz]])
    
    C = bmat([[Cx],[Cy],[Cz]])
    
    return R,C

def gs(b,psi):
    bx = b[:len(b)//3]; by = b[len(b)//3:2*len(b)//3]; bz = b[2*len(b)//3:]
    
    r1 =  np.r_[phi_L2 @ D @ (fx_linear(bx,by,bz)*fem_linear + fx_nonlinear(bx,by,bz)*fem_nonlinear -Hjx) - Cx @ psi,
                phi_L2 @ D @ (fy_linear(bx,by,bz)*fem_linear + fy_nonlinear(bx,by,bz)*fem_nonlinear -Hjy) - Cy @ psi,
                phi_L2 @ D @ (fz_linear(bx,by,bz)*fem_linear + fz_nonlinear(bx,by,bz)*fem_nonlinear -Hjz) - Cz @ psi]
    
    r2 = dphix_H1 @ D @ bx +\
         dphiy_H1 @ D @ by +\
         dphiz_H1 @ D @ bz
    
    return r1,r2

def J(b,psi):
    bx = b[:len(b)//3]; by = b[len(b)//3:2*len(b)//3]; bz = b[2*len(b)//3:]
    psix = dphix_H1.T@psi; psiy = dphiy_H1.T@psi; psiz = dphiz_H1.T@psi
    return np.ones(D.size)@ D @(f_linear(bx,by,bz)*fem_linear + f_nonlinear(bx,by,bz)*fem_nonlinear +(-Hjx)*bx +(-Hjy)*by +(-Hjz)*bz -psix*bx -psiy*by -psiz*bz)


R_out, RS = pde.h1.assembleR3(MESH, space = 'P1', faces = 'ambient_face')

b = np.zeros(3*MESH.nt)
psi = np.zeros(MESH.np)

mu = 1e-2
# mu = 1/2
eps_newton = 1e-5
factor_residual = 1/2

tm2 = time.monotonic()
for i in range(maxIter):
    
    tm = time.monotonic()
    
    R,C = gss(b)
    r1,r2 = gs(b,psi)
    
    # A = bmat([[R,-C@RS.T],
    #           [RS@C.T,None]]).tocsc()
    
    # r = np.r_[r1,RS@r2]
    
    # w = sp.linalg.spsolve(A, -r)
    # wb2 = w[:3*MESH.nt]
    # wpsi2 = RS.T@w[3*MESH.nt:]
    
    tm3 = time.monotonic()
    iR = pde.tools.fastBlockInverse(R)    
    itm = time.monotonic()-tm3
    
    AA = RS@C.T@iR@C@RS.T
    # rr = RS@(-C.T@iR@r1+0*r2)
    rr = RS@(-C.T@iR@r1+r2)
    wpsi = RS.T@chol(AA).solve_A(-rr)
    wb = iR@(C@wpsi-r1)
    w = np.r_[RS@wpsi,wb]
    
    # stop
    
    alpha = 1
    
    # ResidualLineSearch
    # for k in range(1000):
    #     if np.linalg.norm(gs(u+alpha*w),np.inf) <= np.linalg.norm(gs(u),np.inf): break
    #     else: alpha = alpha*factor_residual
    
    # AmijoBacktracking
    float_eps = 1e-12; #float_eps = np.finfo(float).eps
    for kk in range(1000):
        if J(b+alpha*wb,psi+alpha*wpsi)-J(b,psi) <= alpha*mu*(r2@wpsi)+ np.abs(J(b,psi))*float_eps: break
        else: alpha = alpha*factor_residual
        # print(J(b+alpha*wb,psi+alpha*wpsi),J(b,psi),alpha*mu*(r@w),max(abs(w-w_old)))
    
    b_old_i = b
    psi_old_i = psi
    
    b = b + alpha*wb
    psi = psi + alpha*wpsi
    
    print ("NEWTON : %2d " %(i+1)+"||obj: %.2e" %J(b,psi)+"|| ||grad||: %.2e" %np.linalg.norm(np.r_[r1,RS@r2],np.inf)+"||alpha: %.2e" % (alpha) + "|| Step took : %.2f" %(time.monotonic()-tm) + "|| inv took : %.2f" %(itm))
            
    # if ( np.linalg.norm(r,np.inf) < eps_newton):
    #     break
    # if (np.abs(J(b,psi)-J(b_old_i,psi_old_i)) < 1e-8):
    #     break
    
    if (np.abs(J(b,psi)-J(b_old_i,psi_old_i)) < 1e-8*(np.abs(J(b,psi))+np.abs(J(b_old_i,psi_old_i)+1))):
        break
    
elapsed = time.monotonic()-tm2
print('Solving took ', elapsed, 'seconds')

bx = b[:len(b)//3]; by = b[len(b)//3:2*len(b)//3]; bz = b[2*len(b)//3:]

##############################################################################
# Storing to vtk
##############################################################################

grid = pde.tools.vtklib.createVTK(MESH)
pde.tools.vtklib.add_L2_Vector(grid,bx,by,bz,'B_new2')
pde.tools.vtklib.writeVTK(grid, 'magnetostatics_solution2.vtu')
    
# do()