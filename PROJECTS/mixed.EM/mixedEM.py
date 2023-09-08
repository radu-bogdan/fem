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
# import nonlinear_Algorithms
import numba as nb
from scipy.sparse import hstack,vstack,bmat

import matplotlib.pyplot as plt
import matplotlib
cmap = plt.cm.jet

##########################################################################################
# Loading mesh
##########################################################################################
# motor_stator_npz = np.load('meshes/motor_stator.npz', allow_pickle = True)

# p_stator = motor_stator_npz['p'].T
# e_stator = motor_stator_npz['e'].T
# t_stator = motor_stator_npz['t'].T
# q_stator = np.empty(0)
# regions_2d_stator = motor_stator_npz['regions_2d']
# regions_1d_stator = motor_stator_npz['regions_1d']
# m = motor_stator_npz['m']; m_new = m
# j3 = motor_stator_npz['j3']

# motor_rotor_npz = np.load('meshes/motor_rotor.npz', allow_pickle = True)

# p_rotor = motor_rotor_npz['p'].T
# e_rotor = motor_rotor_npz['e'].T
# t_rotor = motor_rotor_npz['t'].T
# q_rotor = np.empty(0)
# regions_2d_rotor = motor_rotor_npz['regions_2d']
# regions_1d_rotor = motor_rotor_npz['regions_1d']

motor_npz = np.load('../meshes/motor.npz', allow_pickle = True)

p = motor_npz['p'].T
e = motor_npz['e'].T
t = motor_npz['t'].T
q = np.empty(0)
regions_2d = motor_npz['regions_2d']
regions_1d = motor_npz['regions_1d']
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']

nu0 = 10**7/(4*np.pi)
MESH = pde.mesh(p,e,t,q)
# MESH.refinemesh()
# MESH.refinemesh()
##########################################################################################



##########################################################################################
# Extract indices
##########################################################################################

ind_stator_outer = MESH.getIndices2d(regions_1d, 'stator_outer', return_index = True)[0]
ind_rotor_outer = MESH.getIndices2d(regions_1d, 'rotor_outer', return_index = True)[0]

ind_edges_rotor_outer = np.where(np.isin(e[:,2],ind_rotor_outer))[0]
edges_rotor_outer = e[ind_edges_rotor_outer,0:2]

ind_air_gap_rotor = MESH.getIndices2d(regions_2d, 'air_gap_rotor', return_index = True)[0]
ind_trig_coils = MESH.getIndices2d(regions_2d, 'coil', return_index = True)[0]
ind_trig_magnets = MESH.getIndices2d(regions_2d, 'magnet', return_index = True)[0]
ind_air_all = MESH.getIndices2d(regions_2d, 'air', return_index = True)[0]
ind_magnet = MESH.getIndices2d(regions_2d, 'magnet', return_index = True)[0]
ind_shaft = MESH.getIndices2d(regions_2d, 'shaft', return_index = True)[0]
ind_coil = MESH.getIndices2d(regions_2d, 'coil', return_index = True)[0]
ind_stator_rotor_and_shaft = MESH.getIndices2d(regions_2d, 'iron', return_index = True)[0]
ind_iron_rotor = MESH.getIndices2d(regions_2d, 'rotor_iron', return_index = True)[0]
ind_rotor_air = MESH.getIndices2d(regions_2d, 'rotor_air', return_index = True)[0]

ind_linear = np.r_[ind_air_all,ind_magnet,ind_shaft,ind_coil]
ind_nonlinear = np.setdiff1d(ind_stator_rotor_and_shaft,ind_shaft)
ind_rotor = np.r_[ind_iron_rotor,ind_magnet,ind_rotor_air,ind_shaft]

R = lambda x: np.array([[np.cos(x),-np.sin(x)],
                        [np.sin(x), np.cos(x)]])

r1 = p[edges_rotor_outer[0,0],0]
a1 = 2*np.pi/edges_rotor_outer.shape[0]

# Adjust points on the outer rotor to be equally spaced.
# for k in range(edges_rotor_outer.shape[0]):
#     p[edges_rotor_outer[k,0],0] = r1*np.cos(a1*(k))
#     p[edges_rotor_outer[k,0],1] = r1*np.sin(a1*(k))
##########################################################################################




##########################################################################################
# Assembling stuff
##########################################################################################

space_Vh = 'N0'
space_Qh = 'P0'
int_order = '2l'

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


C = phi_L2(int_order) @ D(int_order) @ curlphi_Hcurl(int_order).T
Z = sps.csc_matrix((C.shape[0],C.shape[0]))


fem_linear = pde.int.evaluate(MESH, order = int_order, regions = ind_linear).diagonal()
fem_nonlinear = pde.int.evaluate(MESH, order = int_order, regions = ind_nonlinear).diagonal()
fem_rotor = pde.int.evaluate(MESH, order = int_order, regions = ind_rotor).diagonal()
fem_air_gap_rotor = pde.int.evaluate(MESH, order = int_order, regions = ind_air_gap_rotor).diagonal()

Ja = 0; J0 = 0
for i in range(48):
    Ja += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : j3[i], regions = np.r_[ind_trig_coils[i]]).diagonal()
    J0 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : j3[i], regions = np.r_[ind_trig_coils[i]]).diagonal()
Ja = 0*Ja
J0 = 0*J0

M0 = 0; M1 = 0; M00 = 0; M10 = 0
for i in range(16):
    M0 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    M1 += pde.int.evaluate(MESH, order = int_order, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    
    M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()

aM = phix_Hcurl@ D(int_order) @(M0) +\
     phiy_Hcurl@ D(int_order) @(M1)

aJ = phi_L2(int_order)@ D(int_order) @Ja

# iMh = pde.tools.fastBlockInverse(Mh1)
# S = C@iMh@C.T
# r = C@(iMh@aM)

# tm = time.monotonic(); x = sps.linalg.spsolve(S,r); print('dual: ',time.monotonic()-tm)
##########################################################################################



phix_d_Hcurl,phiy_d_Hcurl = pde.hcurl.assemble(MESH, space = 'N0d', matrix = 'phi', order = int_order)
curlphi_d_Hcurl = pde.hcurl.assemble(MESH, space = 'N0d', matrix = 'curlphi', order = int_order)

Md = phix_d_Hcurl @ D(int_order) @ phix_d_Hcurl.T +\
     phiy_d_Hcurl @ D(int_order) @ phiy_d_Hcurl.T
iMd = pde.tools.fastBlockInverse(Md)

Cd = phi_L2(int_order) @ D(int_order) @ curlphi_d_Hcurl.T

aMd = phix_d_Hcurl@ D(int_order) @(M0) +\
      phiy_d_Hcurl@ D(int_order) @(M1)

B0,B1,B2 = pde.hcurl.assembleE(MESH, space = 'N0', matrix = 'M', order = 2)
R0,R1,R2 = pde.hcurl.assembleE(MESH, space = 'N0d', matrix = 'M', order = 2)

phi_e = pde.l2.assembleE(MESH, space = 'P0', matrix = 'M', order = 2)

De = pde.int.assembleE(MESH, order = 2)
KK = phi_e @ De @ (R0+R1+R2).T


# from scipy.sparse import bmat

# SYS = bmat([[Md,Cd.T,KK.T],\
#             [Cd,None,None],
#             [KK,None,None]]).tocsc()


# phi_H1b = pde.hcurl.assembleB(MESH, space = 'N0', matrix = 'M', shape = MESH.NoEdges, order = 4)
# D_stator_outer = pde.int.evaluateB(MESH, order = 4, edges = ind_stator_outer)
# B_stator_outer = phi_H1b@ D_stator_outer @ phi_H1b.T

# phi_H1b2 = pde.hcurl.assembleB(MESH, space = 'N0d', matrix = 'M', shape = 3*MESH.nt, order = 4, edges = np.r_[:MESH.NoEdges])

# stop

##########################################################################################

from nonlinLaws import *

sH = phix_Hcurl.shape[0]
sA = phi_L2_o1.shape[0]

mu0 = (4*np.pi)/10**7
H = 1e-3+np.zeros(sH)
A = 0+np.zeros(sA)

HA = np.r_[H,A]

def gss(allH):
    gxx_H_l  = allH[3];  gxy_H_l  = allH[4];  gyx_H_l  = allH[5];  gyy_H_l  = allH[6];
    gxx_H_nl = allH[10]; gxy_H_nl = allH[11]; gyx_H_nl = allH[12]; gyy_H_nl = allH[13];
    
    gxx_H_Mxx = phix_Hcurl @ D_int_order @ sps.diags(gxx_H_nl*fem_nonlinear + gxx_H_l*fem_linear)@ phix_Hcurl.T
    gyy_H_Myy = phiy_Hcurl @ D_int_order @ sps.diags(gyy_H_nl*fem_nonlinear + gyy_H_l*fem_linear)@ phiy_Hcurl.T
    gxy_H_Mxy = phiy_Hcurl @ D_int_order @ sps.diags(gxy_H_nl*fem_nonlinear + gxy_H_l*fem_linear)@ phix_Hcurl.T
    gyx_H_Myx = phix_Hcurl @ D_int_order @ sps.diags(gyx_H_nl*fem_nonlinear + gyx_H_l*fem_linear)@ phiy_Hcurl.T
    
    M = gxx_H_Mxx + gyy_H_Myy + gxy_H_Mxy + gyx_H_Myx
    
    S = bmat([[M,C.T],\
              [C,None]]).tocsc()
    return S

def gs(allH,A,H):
    gx_H_l  = allH[1]; gy_H_l  = allH[2];
    gx_H_nl = allH[8]; gy_H_nl = allH[9];
    
    r1 = phix_Hcurl @ D_int_order @ (gx_H_l*fem_linear + gx_H_nl*fem_nonlinear + mu0*M0) +\
         phiy_Hcurl @ D_int_order @ (gy_H_l*fem_linear + gy_H_nl*fem_nonlinear + mu0*M1) + C.T@A
    
    # assemble J here!
    # r2 = np.zeros((C.shape[0]))
    r2 = C@H
    
    return np.r_[r1,r2]

def J(allH,H):
    g_H_l = allH[0]; g_H_nl = allH[7];
    return np.ones(D_int_order.size)@ D_int_order @(g_H_l*fem_linear + g_H_nl*fem_nonlinear) + mu0*aM@H


maxIter = 100
epsangle = 1e-5;

angleCondition = np.zeros(5)
eps_newton = 1e-12
factor_residual = 1/2
mu = 0.0001

tm1 = time.monotonic()
for i in range(maxIter):
    
    H = HA[:sH]
    A = HA[sH:]

    Hx = phix_Hcurl.T@H; Hy = phiy_Hcurl.T@H    

    tm = time.monotonic()
    allH = g_nonlinear_all(Hx,Hy)
    gsu = gs(allH,A,H)
    gssu = gss(allH)
    
    print('Evaluating nonlinearity took ', time.monotonic()-tm)
    
    tm = time.monotonic()
    w = sps.linalg.spsolve(gssu,-gsu)
    print('Solving the system took ', time.monotonic()-tm)
    
    norm_w = np.linalg.norm(w)
    norm_gsu = np.linalg.norm(gsu)
    
    if (-(w@gsu)/(norm_w*norm_gsu)<epsangle):
        angleCondition[i%5] = 1
        if np.product(angleCondition)>0:
            w = -gsu
            print("STEP IN NEGATIVE GRADIENT DIRECTION")
    else: angleCondition[i%5]=0
    
    alpha = 1
    
    # ResidualLineSearch
    # for k in range(1000):
        
    #     HAu = HA + alpha*w
    #     Hu = HAu[:2*MESH.NoEdges]
    #     Au = HAu[2*MESH.NoEdges:]
        
    #     Hxu = phix_Hcurl.T@(Hu);
    #     Hyu = phiy_Hcurl.T@(Hu);
        
    #     allHu = g_nonlinear_all(Hxu,Hyu)
        
    #     if np.linalg.norm(gs(allHu,Au,Hu)) <= np.linalg.norm(gs(allH,A,H)): break
    #     else: alpha = alpha*factor_residual
    
    # AmijoBacktracking
    
    tm = time.monotonic()
    float_eps = 1e-8 #np.finfo(float).eps
    for kk in range(1000):
        
        HAu = HA + alpha*w
        Hu = HAu[:sH]; Au = HAu[sH:]
        Hxu = phix_Hcurl.T@(Hu); Hyu = phiy_Hcurl.T@(Hu);
        allHu = g_nonlinear_all(Hxu,Hyu)
        
        
        print(J(allHu,Hu),J(allH,H))
        
        if J(allHu,Hu)-J(allH,H) <= alpha*mu*(gsu@w) + np.abs(J(allH,H))*float_eps: break
        else: alpha = alpha*factor_residual
        
    print('Line search took ', time.monotonic()-tm)
    
    tm = time.monotonic()
    
    HA = HA + alpha*w
    H = HA[:sH]; A = HA[sH:]
    Hx = phix_Hcurl.T@H; Hy = phiy_Hcurl.T@H
    allH = g_nonlinear_all(Hx,Hy)
    
    print('Re-evaluating H took ', time.monotonic()-tm)
    
    
    print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(allH,H)+"|| ||grad||: %2e" %np.linalg.norm(gs(allH,A,H))+"||alpha: %2e" % (alpha)); print("\n")
    if(np.linalg.norm(gs(allH,A,H)) < eps_newton): break

elapsed = time.monotonic()-tm1
print('Solving took ', elapsed, 'seconds')


##########################################################################################
# Post-processing stuff
##########################################################################################

Mhxx = phix_Hcurl @ D_int_order @ phix_Hcurl.T
Mhyy = phiy_Hcurl @ D_int_order @ phiy_Hcurl.T

Mh = Mhxx + Mhyy

phix_Hcurl_o4 = phi_Hcurl(4)[0];
phiy_Hcurl_o4 = phi_Hcurl(4)[1];

Mxx = phix_Hcurl_o4 @ D4 @ phix_Hcurl_o4.T
Myy = phiy_Hcurl_o4 @ D4 @ phiy_Hcurl_o4.T

M = Mxx + Myy

# iMh1 = pde.tools.fastBlockInverse(Mh)

S = vstack((hstack((M, C.T)),
            hstack((C, Z)))).tocsc()

r = np.r_[Mh@H,C@H]

pHA = sps.linalg.spsolve(S, r)
pH = pHA[:sH]

# pde.tools.condest(M)

# H = pH.copy()

##########################################################################################

# phi_Hdiv = lambda x : pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'phi', order = x)
# divphi_Hdiv = lambda x : pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'divphi', order = x)
# phi_L2 = lambda x : pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = x)


# phix_Hdiv_o4 = phi_Hdiv(4)[0];
# phiy_Hdiv_o4 = phi_Hdiv(4)[1];

# phix_Hdiv = phi_Hdiv(int_order)[0]
# phiy_Hdiv = phi_Hdiv(int_order)[1]

# Mdiv_xx = phix_Hdiv_o4 @ D4 @ phix_Hdiv_o4.T
# Mdiv_yy = phiy_Hdiv_o4 @ D4 @ phiy_Hdiv_o4.T

# Mdiv = Mdiv_xx + Mdiv_yy

Hx = phi_Hcurl(1)[0].T@H; Hy = phi_Hcurl(1)[1].T@H
allH = g_nonlinear_all(Hx,Hy)
gx_H_l  = allH[1]; gy_H_l  = allH[2];
gx_H_nl = allH[8]; gy_H_nl = allH[9];

fem_linear = pde.int.evaluate(MESH, order = 1, regions = ind_linear).diagonal()
fem_nonlinear = pde.int.evaluate(MESH, order = 1, regions = ind_nonlinear).diagonal()

# r1 = phix_Hdiv @ D_int_order @ (gx_H_l*fem_linear + gx_H_nl*fem_nonlinear + mu0*M0) +\
#      phiy_Hdiv @ D_int_order @ (gy_H_l*fem_linear + gy_H_nl*fem_nonlinear + mu0*M1)
     
# Cdiv = phi_L2(int_order) @ D(int_order) @ divphi_Hdiv(int_order).T
# Zdiv = sps.csc_matrix((C.shape[0],C.shape[0]))

# S = vstack((hstack((Mdiv, Cdiv.T)),
#             hstack((Cdiv, Zdiv)))).tocsc()

# r = np.r_[r1,0*Cdiv@H]

# pBL = sps.linalg.spsolve(S, r)
# pB = pBL[:sH]

# Bx = phix_Hdiv.T@pB
# By = phiy_Hdiv.T@pB

Bx = (gx_H_l*fem_linear + gx_H_nl*fem_nonlinear)
By = (gy_H_l*fem_linear + gy_H_nl*fem_nonlinear)

# fig = MESH.pdesurf_hybrid(dict(trig = 'P1d',quad = 'Q0',controls = 1), Bx**2+By**2, u_height = 0)
# fig.show()

import plotly.io as pio
pio.renderers.default = "browser"

fig = MESH.pdesurf_hybrid(dict(trig = 'P1', quad = 'Q1', controls = 1), u[:MESH.np], u_height = 1)
fig.show()



# ##########################################################################################

# dphix_H1_o1, dphiy_H1_o1 = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 4)

# Kxx = dphix_H1_o1 @ D(4) @ dphix_H1_o1.T
# Kyy = dphiy_H1_o1 @ D(4) @ dphiy_H1_o1.T

# phi_H1b = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = dphix_H1_o1.shape, order = 4)

# D_stator_outer = pde.int.evaluateB(MESH, order = 4, edges = ind_stator_outer)
# B_stator_outer = phi_H1b@ D_stator_outer @ phi_H1b.T

# aM = dphix_H1_o1@ D(4) @(-M1) +\
#      dphiy_H1_o1@ D(4) @(+M0)
     


# tm = time.monotonic(); x2 = sps.linalg.spsolve(Kxx+Kyy+10**10*B_stator_outer,aM); print('primal: ',time.monotonic()-tm)
    
# ##########################################################################################
# # Plotting stuff
# ##########################################################################################


# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.set_aspect(aspect = 'equal')

# ax1.cla()
# MESH.pdegeom(ax = ax1)
# # MESH.pdemesh2(ax = ax1)

# Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
# chip = ax1.tripcolor(Triang, x, cmap = cmap, shading = 'flat', lw = 0.1)

# ax2 = fig.add_subplot(122)
# ax2.set_aspect(aspect = 'equal')

# ax2.cla()
# MESH.pdegeom(ax = ax2)
# chip = ax2.tripcolor(Triang, x2, cmap = cmap, shading = 'gouraud', lw = 0.1)

# fig.tight_layout()
# fig.show()

