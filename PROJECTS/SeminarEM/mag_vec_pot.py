import sys
sys.path.insert(0,'../../') # adds parent directory
# sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
from sksparse.cholmod import cholesky as chol
import plotly.io as pio
pio.renderers.default = 'browser'
import nonlinear_Algorithms
import numba as nb


##########################################################################################
# Loading mesh
##########################################################################################
motor_npz = np.load('meshes/motor.npz', allow_pickle = True)

p = motor_npz['p'].T
e = motor_npz['e'].T
t = motor_npz['t'].T
q = np.empty(0)
regions_2d = motor_npz['regions_2d']
regions_1d = motor_npz['regions_1d']
m = motor_npz['m']
j3 = motor_npz['j3']

#KORREKTUR !
# BR = 0
##########################################################################################



##########################################################################################
# Parameters
##########################################################################################

ORDER = 1

nu0 = 10**7/(4*np.pi)
m = m*nu0*1.158095238095238


MESH = pde.mesh(p,e,t,q)
# MESH.refinemesh()
# MESH.refinemesh()
##########################################################################################



##########################################################################################
# Extract indices
##########################################################################################

def getIndices(liste,name,exact = 0,return_index=False):
    if exact == 0:
        ind = np.flatnonzero(np.core.defchararray.find(list(liste),name)!=-1)
    else:
        ind = [i for i, x in enumerate(list(liste)) if x == name][0]
    elem = np.where(np.isin(MESH.t[:,3],ind))[0]
    mask = np.zeros(MESH.nt); mask[elem] = 1
    if return_index:
        return ind, elem, mask
    else:
        return elem, mask

trig_air_all, mask_air_all = getIndices(regions_2d,'air')
trig_stator_rotor_and_shaft, mask_stator_rotor_and_shaft = getIndices(regions_2d,'iron')
trig_magnet, mask_magnet = getIndices(regions_2d,'magnet')
trig_coil, mask_coil = getIndices(regions_2d,'coil')
trig_shaft, mask_shaft = getIndices(regions_2d,'shaft')

trig_stator_rotor = np.setdiff1d(trig_stator_rotor_and_shaft,trig_shaft)
mask_stator_rotor = np.zeros(MESH.nt); mask_stator_rotor[trig_stator_rotor]=1

mask_linear    = mask_air_all + mask_magnet + mask_shaft + mask_coil
mask_nonlinear = mask_stator_rotor

trig_linear = np.r_[trig_air_all,trig_magnet,trig_shaft,trig_coil]
trig_nonlinear = trig_stator_rotor

ind_stator_outer = np.flatnonzero(np.core.defchararray.find(list(regions_1d),'stator_outer')!=-1)
edges_stator_outer = np.where(np.isin(e[:,2],ind_stator_outer))[0]

ind_trig_coils = {}
for i in range(48):
    ind_trig_coils[i] = getIndices(regions_2d, 'coil' + str(i+1), exact = 1, return_index = True)[0]

ind_trig_magnets = {}
for i in range(16):
    ind_trig_magnets[i] = getIndices(regions_2d, 'magnet' + str(i+1), exact = 1, return_index = True)[0]


##########################################################################################



##########################################################################################
# Order configuration
##########################################################################################
if ORDER == 1:
    poly = 'P1'
    dxpoly = 'P0'
    order_phiphi = 2
    order_dphidphi = 0
    new_mask_linear = mask_linear
    new_mask_nonlinear = mask_nonlinear
    
if ORDER == 2:
    poly = 'P2'
    dxpoly = 'P1'
    order_phiphi = 4
    order_dphidphi = 2
    new_mask_linear = np.tile(mask_linear,(3,1)).T.flatten()
    new_mask_nonlinear = np.tile(mask_nonlinear,(3,1)).T.flatten()
##########################################################################################



##########################################################################################
# Brauer/Nonlinear laws ... ?
##########################################################################################

k1 = 49.4; k2 = 1.46; k3 = 520.6

f_iron = lambda x,y : k1/(2*k2)*(np.exp(k2*(x**2+y**2))-1) + 1/2*k3*(x**2+y**2) # magnetic energy density in iron

# c = 6.63
# nu = lambda x,y : ((x**2+y**2)<c)*(k1*np.exp(k2*(x**2+y**2))+k3) + nu0*((x**2+y**2)>c)
# nux = lambda x,y : ((x**2+y**2)<c)*(2*x*k1*k2*np.exp(k2*(x**2+y**2))) + 0*((x**2+y**2)>c)
# nuy = lambda x,y : ((x**2+y**2)<c)*(2*y*k1*k2*np.exp(k2*(x**2+y**2))) + 0*((x**2+y**2)>c)

nu = lambda x,y : k1*np.exp(k2*(x**2+y**2))+k3
nux = lambda x,y : 2*x*k1*k2*np.exp(k2*(x**2+y**2))
nuy = lambda x,y : 2*y*k1*k2*np.exp(k2*(x**2+y**2))

fx_iron = lambda x,y : nu(x,y)*x
fy_iron = lambda x,y : nu(x,y)*y
fxx_iron = lambda x,y : nu(x,y) + x*nux(x,y)
fxy_iron = lambda x,y : x*nuy(x,y)
fyx_iron = lambda x,y : y*nux(x,y)
fyy_iron = lambda x,y : nu(x,y) + y*nuy(x,y)

f_linear = lambda x,y : 1/2*nu0*(x**2+y**2)
fx_linear = lambda x,y : nu0*x
fy_linear = lambda x,y : nu0*y
fxx_linear = lambda x,y : nu0 + 0*x
fxy_linear = lambda x,y : x*0
fyx_linear = lambda x,y : y*0
fyy_linear = lambda x,y : nu0 + 0*y


f   = lambda ux,uy :   f_linear(ux,uy)*new_mask_linear +   f_iron(ux,uy)*new_mask_nonlinear
fx  = lambda ux,uy :  fx_linear(ux,uy)*new_mask_linear +  fx_iron(ux,uy)*new_mask_nonlinear
fy  = lambda ux,uy :  fy_linear(ux,uy)*new_mask_linear +  fy_iron(ux,uy)*new_mask_nonlinear
fxx = lambda ux,uy : fxx_linear(ux,uy)*new_mask_linear + fxx_iron(ux,uy)*new_mask_nonlinear
fxy = lambda ux,uy : fxy_linear(ux,uy)*new_mask_linear + fxy_iron(ux,uy)*new_mask_nonlinear
fyx = lambda ux,uy : fyx_linear(ux,uy)*new_mask_linear + fyx_iron(ux,uy)*new_mask_nonlinear
fyy = lambda ux,uy : fyy_linear(ux,uy)*new_mask_linear + fyy_iron(ux,uy)*new_mask_nonlinear
##########################################################################################



##########################################################################################
# Assembling stuff
##########################################################################################

tm = time.monotonic()

phi_H1  = pde.h1.assemble(MESH, space = poly, matrix = 'M', order = order_phiphi)
dphix_H1, dphiy_H1 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_dphidphi)
dphix_H1_o0, dphiy_H1_o0 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = 0)
dphix_H1_o1, dphiy_H1_o1 = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = 1)
dphix_H1_order_phiphi, dphiy_H1_order_phiphi = pde.h1.assemble(MESH, space = poly, matrix = 'K', order = order_phiphi)
phi_H1b = pde.h1.assembleB(MESH, space = poly, matrix = 'M', shape = phi_H1.shape, order = order_phiphi)
phi_L2 = pde.l2.assemble(MESH, space = dxpoly, matrix = 'M', order = order_dphidphi)


D_order_dphidphi = pde.int.assemble(MESH, order = order_dphidphi)
D_order_phiphi = pde.int.assemble(MESH, order = order_phiphi)
D_order_phiphi_b = pde.int.assembleB(MESH, order = order_phiphi)


Kxx = dphix_H1 @ D_order_dphidphi @ dphix_H1.T
Kyy = dphiy_H1 @ D_order_dphidphi @ dphiy_H1.T
Cx = phi_L2 @ D_order_dphidphi @ dphix_H1.T
Cy = phi_L2 @ D_order_dphidphi @ dphiy_H1.T

D_stator_outer = pde.int.evaluateB(MESH, order = order_phiphi, edges = ind_stator_outer)
B_stator_outer = phi_H1b@ D_stator_outer @ phi_H1b.T

penalty = 1

J = 0; J0 = 0
for i in range(48):
    J += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : j3[i], regions = np.r_[ind_trig_coils[i]]).diagonal()
    J0+= pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : j3[i], regions = np.r_[ind_trig_coils[i]]).diagonal()

M0 = 0; M1 = 0; M00 = 0
for i in range(16):
    M0 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    M1 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    
    M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()

aJ = 0*phi_H1@ D_order_phiphi @J

aM = dphix_H1_order_phiphi@ D_order_phiphi @(-M1) +\
     dphiy_H1_order_phiphi@ D_order_phiphi @(+M0)

aMnew = aM

# fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1), M00, u_height=0)
# fig.show()

def update_left(ux,uy):
    fxx_grad_u_Kxx = dphix_H1 @ D_order_dphidphi @ sps.diags(fxx(ux,uy))@ dphix_H1.T
    fyy_grad_u_Kyy = dphiy_H1 @ D_order_dphidphi @ sps.diags(fyy(ux,uy))@ dphiy_H1.T
    fxy_grad_u_Kxy = dphiy_H1 @ D_order_dphidphi @ sps.diags(fxy(ux,uy))@ dphix_H1.T
    fyx_grad_u_Kyx = dphix_H1 @ D_order_dphidphi @ sps.diags(fyx(ux,uy))@ dphiy_H1.T
    return (fxx_grad_u_Kxx + fyy_grad_u_Kyy + fxy_grad_u_Kxy + fyx_grad_u_Kyx) + penalty*B_stator_outer
  
def update_right(u,ux,uy):
    # return -Cx.T @ fx(ux,uy) -Cy.T @ fy(ux,uy) -penalty*B_stator_outer@u + aJ - aM
    return -dphix_H1 @ D_order_dphidphi @ fx(ux,uy) -dphiy_H1 @ D_order_dphidphi @ fy(ux,uy) -penalty*B_stator_outer@u + aJ - aM


def g(u):
    ux = dphix_H1.T@u
    uy = dphiy_H1.T@u
    return -update_right(u,ux,uy)

def h(u):
    ux = dphix_H1.T@u
    uy = dphiy_H1.T@u
    return update_left(ux, uy)

def f(u):
    return 0#f(ux,uy)

# Solving h(un)w = g(un), un = un + w

print('Assembling + stuff ', time.monotonic()-tm)
##########################################################################################



##########################################################################################
# Solving with Newton
##########################################################################################

def ResidualLineSearch(dJsx,factor=1/2,mu=0.01):
    alpha = 1
    for i in range(1000):
        if np.linalg.norm(dJsx(alpha))<=np.linalg.norm(dJsx(0)):
            return alpha
        else:
            alpha=alpha*factor
    return alpha

u = 0+np.zeros(shape = Kxx.shape[0])

# tm = time.monotonic()
# np.seterr(all='ignore')
# u = nonlinear_Algorithms.NewtonSparse(f, g, h, x0 = u, use_chol = 2, maxIter = 100, printoption = 1)[0]
# np.seterr(all='warn')
# print('Solving took ', time.monotonic()-tm, 'seconds')

maxIter = 100
epsangle = 1e-5;
u = np.zeros(shape = Kxx.shape[0])
angleCondition = np.zeros(5)
eps = 1e-8


def fss(u):
    ux = dphix_H1.T@u
    uy = dphiy_H1.T@u
    return update_left(ux, uy)

def fs(u):
    ux = dphix_H1.T@u
    uy = dphiy_H1.T@u
    return update_right(u,ux,uy)

def f(u):
    return 0

tm = time.monotonic()
for i in range(maxIter):
    fsu = fs(u)
    w = chol(fss(u)).solve_A(fsu)
    
    norm_w = np.linalg.norm(w)
    norm_fsu = np.linalg.norm(fsu)
    
    if ((w@fsu)/(norm_w*norm_fsu)<epsangle):
        angleCondition[i%5] = 1
        if np.product(angleCondition)>0:
            w = fsu
            print("STEP IN NEGATIVE GRADIENT DIRECTION")
    else:
        angleCondition[i%5]=0
    
    fsw = lambda alpha: fs(u+alpha*w)
    alpha = ResidualLineSearch(fsw)
    u = u + alpha*w
    print ("NEWTON: Iteration: %2d" %(i+1)+"||obj: %2e" % (f(u))+"|| ||grad||: %2e" % (np.linalg.norm(fs(u)))+"||alpha: %2e" % (alpha))
    
    if(np.linalg.norm(np.linalg.norm(fs(u)))<eps):
        break
    
print('Solving took ', time.monotonic()-tm, 'seconds')
    
    
    # dJx=dJ(x)
    # sx=solve_ddJ(x,-dJx)
    # # sx = sps.linalg.spsolve(ddJ(´x),-dJx)
    # if (np.dot(sx,-dJx)/np.linalg.norm(sx)/np.linalg.norm(dJx)<epsangle):
    #     angleCondition[i%5]=1      
    #     if np.product(angleCondition)>0:
    #         sx=-dJx
    #         print("STEP IN NEGATIVE GRADIENT DIRECTION")
    # else:
    #     angleCondition[i%5]=0
    # #F = lambda alpha:J(x+alpha*sx)
    # #dF = lambda alpha: np.dot(dJ(x+alpha*sx),sx)
    # dJsx= lambda alpha: dJ(x+alpha*sx)
    # #alpha=WolfePowell(F,dF)
    # # alpha=AmijoBacktracking(F, dF)
    # alpha=ResidualLinesearch(dJsx)
    # x=x+alpha*sx
    # if printoption:
    #     print ("NEWTON: Iteration: %2d" %(i+1)+"||obj: %2e" % (J(x))+"|| ||grad||: %2e" % (np.linalg.norm(dJ(x)))+"||alpha: %2e" % (alpha))
    # if(np.linalg.norm(np.linalg.norm(dJ(x)))<eps):
    #     flag=1
    #     return x,flag
    
    
    
    # print(np.linalg.norm(w),
    #       np.linalg.norm(ux),
    #       np.linalg.norm(uy))
    
    # if i%10==0:
    #     fig = MESH.pdesurf_hybrid(dict(trig = 'P1', quad = 'Q1', controls = 1), u[:MESH.np], u_height = 0)
    #     fig.layout.scene.camera.projection.type = "orthographic"
    #     fig.show()




# K = Kxx + Kyy + penalty*B_stator_outer
# u = chol(K).solve_A(aJ-aM)

if dxpoly == 'P1':
    ux = dphix_H1_o1.T@u
    uy = dphiy_H1_o1.T@u
    norm_ux = np.sqrt(ux**2+uy**2)
    fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', quad = 'Q1d', controls = 1), norm_ux, u_height = 0)
    fig.data[0].colorscale='Jet'
    
if dxpoly == 'P0':
    ux = dphix_H1_o0.T@u
    uy = dphiy_H1_o0.T@u
    norm_ux = np.sqrt(ux**2+uy**2)
    fig = MESH.pdesurf_hybrid(dict(trig = 'P0', quad = 'Q0', controls = 1), norm_ux, u_height = 0)
    fig.data[0].colorscale='Jet'
    
# print(norm_ux.max(),norm_ux.min())    
fig.show()
##########################################################################################






# fig = MESH.pdesurf_hybrid(dict(trig = 'P1', quad = 'Q1', controls = 1), u[:MESH.np], u_height = 1)
# fig.layout.scene.camera.projection.type = "orthographic"
# fig.show()

