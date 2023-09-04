import sys
sys.path.insert(0,'../../') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import plotly.io as pio
pio.renderers.default = 'browser'
import ngsolve as ng

import numba as nb
# import pyamg
from scipy.sparse import hstack,vstack,bmat
from sksparse.cholmod import cholesky as chol

import matplotlib.pyplot as plt
import matplotlib
cmap = plt.cm.jet

##########################################################################################
# Loading mesh
##########################################################################################

motor_npz = np.load('../meshes/motor.npz', allow_pickle = True)

# p = motor_npz['p']
# e = motor_npz['e']
# t = motor_npz['t']
# q = np.empty(0)
# regions_2d = motor_npz['regions_2d']
# regions_1d = motor_npz['regions_1d']

# MESH = pde.mesh(p,e,t,q,regions_2d,regions_1d)
# MESH.refinemesh()
# MESH.refinemesh()

geoOCC = motor_npz['geoOCC'].tolist()
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']


geoOCCmesh = geoOCC.GenerateMesh()
ngsolve_mesh = ng.Mesh(geoOCCmesh)
ngsolve_mesh.Refine()
ngsolve_mesh.Refine()
# ngsolve_mesh.Refine()


nu0 = 10**7/(4*np.pi)
MESH = pde.mesh.netgen(ngsolve_mesh.ngmesh)

linear = '*air,*magnet,shaft_iron,*coil'
nonlinear = 'stator_iron,rotor_iron'
rotor = 'rotor_iron,*magnet,rotor_air,shaft_iron'

# a1 = 0.00320570678937734
# r1 = 0.07863225
# ind_edges_rotor_outer = np.where(np.isin(e[:,2],593))[0]
# edges_rotor_outer = e[ind_edges_rotor_outer,0:2]
# for k in range(edges_rotor_outer.shape[0]):
#     p[edges_rotor_outer[k,0],0] = r1*np.cos(a1*(k))
#     p[edges_rotor_outer[k,0],1] = r1*np.sin(a1*(k))

##########################################################################################


##########################################################################################
# Assembling stuff
##########################################################################################

space_Vh = 'N1'
space_Qh = 'P1'
space_Vhd = 'N1d'
# space_Qh = 'P1_orth_divN1'
int_order = 4

tm = time.monotonic()

phi_Hcurl = lambda x : pde.hcurl.assemble(MESH, space = space_Vh, matrix = 'phi', order = x)
curlphi_Hcurl = lambda x : pde.hcurl.assemble(MESH, space = space_Vh, matrix = 'curlphi', order = x)
phi_L2 = lambda x : pde.l2.assemble(MESH, space = space_Qh, matrix = 'M', order = x)

D = lambda x : pde.int.assemble(MESH, order = x)

Mh = lambda x: phi_Hcurl(x)[0] @ D(x) @ phi_Hcurl(x)[0].T + \
               phi_Hcurl(x)[1] @ D(x) @ phi_Hcurl(x)[1].T

D1 = D(1); D2 = D(2); D4 = D(4); Mh1 = Mh(1); Mh2 = Mh(2)
D_int_order = D(int_order)

phi_L2_o1 = phi_L2(1)
curlphi_Hcurl_o1 = curlphi_Hcurl(1)

phix_Hcurl = phi_Hcurl(int_order)[0];
phiy_Hcurl = phi_Hcurl(int_order)[1];


inv = lambda x : pde.tools.fastBlockInverse(x)

C = phi_L2(int_order) @ D(int_order) @ curlphi_Hcurl(int_order).T


fem_linear = pde.int.evaluate(MESH, order = int_order, regions = linear).diagonal()
fem_nonlinear = pde.int.evaluate(MESH, order = int_order, regions = nonlinear).diagonal()
fem_rotor = pde.int.evaluate(MESH, order = int_order, regions = rotor).diagonal()
fem_air_gap_rotor = pde.int.evaluate(MESH, order = int_order, regions = 'air_gap_rotor').diagonal()

Ja = 0; J0 = 0
for i in range(48):
    Ja += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : j3[i], regions = 'coil'+str(i+1)).diagonal()
    J0 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : j3[i], regions = 'coil'+str(i+1)).diagonal()
Ja = 0*Ja
J0 = 0*J0

M0 = 0; M1 = 0; M00 = 0; M10 = 0
for i in range(16):
    M0 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
    M1 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()
    
    M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = 'magnet'+str(i+1)).diagonal()
    M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = 'magnet'+str(i+1)).diagonal()

# aM = phix_Hcurl@ D(int_order) @(M0) +\
#      phiy_Hcurl@ D(int_order) @(M1)

# aJ = phi_L2(int_order)@ D(int_order) @Ja

# iMh = pde.tools.fastBlockInverse(Mh1)
# S = C@iMh@C.T
# r = C@(iMh@aM)

# tm = time.monotonic(); x = sps.linalg.spsolve(S,r); print('dual: ',time.monotonic()-tm)
##########################################################################################


phix_d_Hcurl,phiy_d_Hcurl = pde.hcurl.assemble(MESH, space = space_Vhd, matrix = 'phi', order = int_order)
curlphi_d_Hcurl = pde.hcurl.assemble(MESH, space = space_Vhd, matrix = 'curlphi', order = int_order)

Md = phix_d_Hcurl @ D(int_order) @ phix_d_Hcurl.T +\
     phiy_d_Hcurl @ D(int_order) @ phiy_d_Hcurl.T
iMd = pde.tools.fastBlockInverse(Md)

iMd.data = iMd.data*(np.abs(iMd.data)>1e-7)
iMd.eliminate_zeros()

Cd = phi_L2(int_order) @ D(int_order) @ curlphi_d_Hcurl.T

aMd = phix_d_Hcurl@ D(int_order) @(M0) +\
      phiy_d_Hcurl@ D(int_order) @(M1)

R0,R1,R2 = pde.hcurl.assembleE(MESH, space = space_Vhd, matrix = 'M', order = 2)

phi_e = pde.l2.assembleE(MESH, space = 'P1', matrix = 'M', order = 1)

De = pde.int.assembleE(MESH, order = 1)
KK = phi_e @ De @ (R0+R1+R2).T

KK = KK[np.r_[2*MESH.NonSingle_Edges,\
              2*MESH.NonSingle_Edges+1],:]

# KK = KK[MESH.NonSingle_Edges,:]

##########################################################################################

# from nonlinLaws import *
from nonlinLaws import *

sH = phix_d_Hcurl.shape[0]
sA = phi_L2_o1.shape[0]
sL = KK.shape[0]

mu0 = (4*np.pi)/10**7
H = 1e-2+np.zeros(sH)
A = 0+np.zeros(sA)
L = 0+np.zeros(sL)

HAL = np.r_[H,A,L]


def gs_hybrid(allH,A,H,L):
    gx_H_l  = allH[1]; gy_H_l  = allH[2];
    gx_H_nl = allH[8]; gy_H_nl = allH[9];
    
    r1 = phix_d_Hcurl @ D_int_order @ (gx_H_l*fem_linear + gx_H_nl*fem_nonlinear + mu0*M0) +\
         phiy_d_Hcurl @ D_int_order @ (gy_H_l*fem_linear + gy_H_nl*fem_nonlinear + mu0*M1) + Cd.T@A +KK.T@L
    r2 = Cd@H
    r3 = KK@H
    
    # r = -KK@iMd@r1 + KK@iMd@Cd.T@iBBd@(Cd@iMd@r1-r2)
    
    return np.r_[r1,r2,r3]

def J(allH,H):
    g_H_l = allH[0]; g_H_nl = allH[7];
    return np.ones(D_int_order.size)@ D_int_order @(g_H_l*fem_linear + g_H_nl*fem_nonlinear) + mu0*aMd@H


maxIter = 100
epsangle = 1e-5;

angleCondition = np.zeros(5)
eps_newton = 1e-12
factor_residual = 1/2
mu = 0.0001

tm1 = time.monotonic()
for i in range(maxIter):
    
    H = HAL[:sH]
    A = HAL[sH:sH+sA]
    L = HAL[sH+sA:]
    
    ##########################################################################################
    Hx = phix_d_Hcurl.T@H; Hy = phiy_d_Hcurl.T@H
    print(Hx.max(),Hy.max())
    
    tm = time.monotonic()
    allH = g_nonlinear_all(Hx,Hy)
    
    print('Evaluating nonlinearity took ', time.monotonic()-tm)
    
    tm = time.monotonic()
    gxx_H_l  = allH[3];  gxy_H_l  = allH[4];  gyx_H_l  = allH[5];  gyy_H_l  = allH[6];
    gxx_H_nl = allH[10]; gxy_H_nl = allH[11]; gyx_H_nl = allH[12]; gyy_H_nl = allH[13];
    
    gxx_H_Mxx = phix_d_Hcurl @ D_int_order @ sps.diags(gxx_H_nl*fem_nonlinear + gxx_H_l*fem_linear)@ phix_d_Hcurl.T
    gyy_H_Myy = phiy_d_Hcurl @ D_int_order @ sps.diags(gyy_H_nl*fem_nonlinear + gyy_H_l*fem_linear)@ phiy_d_Hcurl.T
    gxy_H_Mxy = phiy_d_Hcurl @ D_int_order @ sps.diags(gxy_H_nl*fem_nonlinear + gxy_H_l*fem_linear)@ phix_d_Hcurl.T
    gyx_H_Myx = phix_d_Hcurl @ D_int_order @ sps.diags(gyx_H_nl*fem_nonlinear + gyx_H_l*fem_linear)@ phiy_d_Hcurl.T
    
    Md = gxx_H_Mxx + gyy_H_Myy + gxy_H_Mxy + gyx_H_Myx
    
    gx_H_l  = allH[1]; gy_H_l  = allH[2]; gx_H_nl = allH[8]; gy_H_nl = allH[9];
    
    r1 = phix_d_Hcurl @ D_int_order @ (gx_H_l*fem_linear + gx_H_nl*fem_nonlinear + mu0*M0) +\
         phiy_d_Hcurl @ D_int_order @ (gy_H_l*fem_linear + gy_H_nl*fem_nonlinear + mu0*M1) + Cd.T@A + KK.T@L
    r2 = Cd@H
    r3 = KK@H
    
    print('Assembling took ', time.monotonic()-tm)
    
    
    # print(sps.linalg.eigs(Md,which = 'LR',k=2)[0])
    
    tm = time.monotonic()
    iMd = inv(Md)
    iBBd = inv(Cd@iMd@Cd.T)
    print('Inverting took ', time.monotonic()-tm)
    
    
    
    tm = time.monotonic()
    gssuR = -KK@iMd@KK.T + KK@iMd@Cd.T@iBBd@Cd@iMd@KK.T
    gsuR = -(KK@iMd@r1-r3) + KK@iMd@Cd.T@iBBd@(Cd@iMd@r1-r2)
    print('Multiplication took ', time.monotonic()-tm)
    
    
    
    tm = time.monotonic()
    wL = chol(-gssuR).solve_A(gsuR)
    print('Solving the system took ', time.monotonic()-tm)
    
    # tm = time.monotonic()
    # wL = sps.linalg.spsolve(-gssuR,gsuR)
    # print('Solving the system took ', time.monotonic()-tm)
    
    wA = iBBd@Cd@iMd@(-r1-KK.T@wL)+iBBd@r2
    wH = iMd@(-Cd.T@wA-KK.T@wL-r1)
    w = np.r_[wH,wA,wL]
    
    gsu = gs_hybrid(allH,A,H,L)
    
    ##########################################################################################
    
    
    
    norm_w = np.linalg.norm(w)
    norm_gsu = np.linalg.norm(gsu)
    
    if (-(w@gsu)/(norm_w*norm_gsu)<epsangle):
        angleCondition[i%5] = 1
        if np.product(angleCondition)>0:
            w = -gsu
            print("STEP IN NEGATIVE GRADIENT DIRECTION")
            break
    else: angleCondition[i%5]=0
    
    alpha = 1
    
    # ResidualLineSearch
    # for k in range(1000):
        
    #     HALu = HAL + alpha*w
    #     Hu = HALu[:sH]
    #     Au = HALu[sH:sH+sA]
    #     Lu = HALu[sH+sA:]
        
    #     Hxu = phix_d_Hcurl.T@(Hu)
    #     Hyu = phiy_d_Hcurl.T@(Hu)
        
    #     allHu = g_nonlinear_all(Hxu,Hyu)
        
    #     if np.linalg.norm(gs_hybrid(allHu,Au,Hu,Lu)) <= np.linalg.norm(gs_hybrid(allH,A,H,L)): break
    #     else: alpha = alpha*factor_residual
    
    # AmijoBacktracking
    
    tm = time.monotonic()
    float_eps = 1e-8 #np.finfo(float).eps
    for kk in range(1000):
        
        HALu = HAL + alpha*w
        Hu = HALu[:sH]
        
        Hxu = phix_d_Hcurl.T@Hu; Hyu = phiy_d_Hcurl.T@Hu;
        allHu = g_nonlinear_all(Hxu,Hyu)
        
        # print(J(allHu,Hu),J(allH,H),J(allHu,Hu)-J(allH,H))
        
        if J(allHu,Hu)-J(allH,H) <= alpha*mu*(gsu@w) + np.abs(J(allH,H))*float_eps: break
        else: alpha = alpha*factor_residual
        
    print('Line search took ', time.monotonic()-tm)
    
    HAL = HALu; H = Hu; Hx = Hxu; Hy = Hyu; allH = allHu
    
    print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(allH,H)+"|| ||grad||: %2e" %np.linalg.norm(gs_hybrid(allH,A,H,L),np.inf)+"||alpha: %2e" % (alpha)); print("\n")
    if(np.linalg.norm(gs_hybrid(allH,A,H,L),np.inf) < eps_newton): break

elapsed = time.monotonic()-tm1
print('Solving took ', elapsed, 'seconds')


##########################################################################################
# Torque
##########################################################################################

fem_linear = pde.int.evaluate(MESH, order = int_order, regions = linear).diagonal()
fem_nonlinear = pde.int.evaluate(MESH, order = int_order, regions = nonlinear).diagonal()

#outer radius rotor
r_outer = 78.63225*10**(-3);
#sliding mesh rotor
r_sliding = 78.8354999*10**(-3);
#sliding mesh stator
r_sliding2 = 79.03874999*10**(-3);
#inner radius stator
r_inner = 79.242*10**(-3);

r1 = r_outer
r2 = r_inner

# dazwischen = lambda x,y : 
scale = lambda x,y : 1*(x**2+y**2<r1**2)+(x**2+y**2-r2**2)/(r1**2-r2**2)*(x**2+y**2>r1**2)*(x**2+y**2<r2**2)
scalex = lambda x,y : (2*x)/(r1**2-r2**2)#*(x**2+y**2>r1**2)*(x**2+y**2<r2**2)
scaley = lambda x,y : (2*y)/(r1**2-r2**2)#*(x**2+y**2>r1**2)*(x**2+y**2<r2**2)

v = lambda x,y : np.r_[-y,x]*scale(x,y) # v wird nie wirklich gebraucht...

v1x = lambda x,y : -y*scalex(x,y)
v1y = lambda x,y : -scale(x,y)-y*scaley(x,y)
v2x = lambda x,y :  scale(x,y)+x*scalex(x,y)
v2y = lambda x,y :  x*scaley(x,y)

# ind_air_gaps = MESH.getIndices2d(regions_2d, 'air_gap', exact = 0, return_index = True)[0]

v1x_fem = pde.int.evaluate(MESH, order = int_order, coeff = v1x, regions = '*air_gap').diagonal()
v1y_fem = pde.int.evaluate(MESH, order = int_order, coeff = v1y, regions = '*air_gap').diagonal()
v2x_fem = pde.int.evaluate(MESH, order = int_order, coeff = v2x, regions = '*air_gap').diagonal()
v2y_fem = pde.int.evaluate(MESH, order = int_order, coeff = v2y, regions = '*air_gap').diagonal()

scale_fem = pde.int.evaluate(MESH, order = int_order, coeff = scale, regions = '*air_gap,'+rotor).diagonal()
one_fem = pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : 1+0*x+0*y, regions = '*air_gap,'+rotor).diagonal()

Hx = phix_d_Hcurl.T@H; Hy = phiy_d_Hcurl.T@H
gx_H_l  = allH[1]; gy_H_l  = allH[2]; gx_H_nl = allH[8]; gy_H_nl = allH[9]; g_H_l = allH[0]; g_H_nl = allH[7];

gHx = gx_H_l*fem_linear + gx_H_nl*fem_nonlinear + mu0*M0
gHy = gy_H_l*fem_linear + gy_H_nl*fem_nonlinear + mu0*M1

gH = g_H_l*fem_linear + g_H_nl*fem_nonlinear

# b1 = uy
# b2 = -ux
# fu = f(ux,uy)+1/2*(M0_dphi*uy-M1_dphi*ux)
# fbb1 =  fy(ux,uy)+M0_dphi
# fbb2 = -fx(ux,uy)+M1_dphi
# a_Pk = u_Pk

# term1 = (fu + fbb1*b1 +fbb2*b2 -Ja0*a_Pk)*(v1x_fem + v2y_fem)

# term2 = (fbb1*b1)*v1x_fem + \
#         (fbb2*b1)*v2x_fem + \
#         (fbb1*b2)*v1y_fem + \
#         (fbb2*b2)*v2y_fem


term1 = (gH +gHx*Hx +gHy*Hy)*(v1x_fem + v2y_fem)
term2 = (gHx*Hx)*v1x_fem + \
        (gHy*Hx)*v2x_fem + \
        (gHx*Hy)*v1y_fem + \
        (gHy*Hy)*v2y_fem


term_2 = -(term1+term2)
tor = one_fem@D_int_order@term_2
print('Torque:', tor)




##########################################################################################

phix_d_Hcurl,phiy_d_Hcurl = pde.hcurl.assemble(MESH, space = space_Vhd, matrix = 'phi', order = 1)

Hx = phix_d_Hcurl.T@H; Hy = phiy_d_Hcurl.T@H
allH = g_nonlinear_all(Hx,Hy)
gx_H_l  = allH[1]; gy_H_l  = allH[2];
gx_H_nl = allH[8]; gy_H_nl = allH[9];

fem_linear = pde.int.evaluate(MESH, order = 1, regions = linear).diagonal()
fem_nonlinear = pde.int.evaluate(MESH, order = 1, regions = nonlinear).diagonal()

Bx = (gx_H_l*fem_linear + gx_H_nl*fem_nonlinear)
By = (gy_H_l*fem_linear + gy_H_nl*fem_nonlinear)

fig = MESH.pdesurf(Bx**2+By**2)
fig.show()

##########################################################################################

# fig = plt.figure()
# fig.show()
# ax1 = fig.add_subplot(111)
# chip = ax1.tripcolor(Triang, Bx**2+By**2, cmap = cmap, shading = 'flat', lw = 0.1)