#!/usr/bin/python --relpath_append ../

import sys
sys.path.insert(0,'..') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import geometries

import plotly.io as pio
pio.renderers.default = 'browser'

np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = np.inf)
np.set_printoptions(precision = 8)

d = 3
l = 10
gmsh.initialize()
gmsh.model.add("Capacitor plates")
geometries.capacitorPlates(a = 20,b = 20,c = 0.5,d = d,l = l)
gmsh.option.setNumber("Mesh.Algorithm", 2)
gmsh.option.setNumber("Mesh.MeshSizeMax", 1)
# gmsh.fltk.run()
p,e,t,q = pde.petq_generate()

MESH = pde.initmesh(p,e,t,q)

# TODO:  MESH = pde.refinemesh(p,e,t,q)
BASIS = pde.basis()
LISTS = pde.lists(MESH)

g1 = lambda x,y : -1+0*x
g2 = lambda x,y :  1+0*x

Kxx, Kyy, Kxy, Kyx = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'K'))
M = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'M'))

sizeM = M.shape[0]

walls = np.r_[5,6,7,8,9,10,11,12]
left_block = np.r_[5,6,7,8]
right_block = np.r_[9,10,11,12]

B_full = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P1', size = sizeM))
B_left_block = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P1', size = sizeM, edges = left_block))
B_right_block = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P1', size = sizeM, edges = right_block))
B_walls = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P1', size = sizeM, edges = walls))
# M_f = pde.projections.assem(MESH, BASIS, LISTS, dict(trig = 'P1'), f_rhs)
Cx,Cy = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'C'))

B_g  = pde.projections.assem_H1_b(MESH, BASIS, LISTS, dict(space = 'P1', order = 2, edges = left_block, size = sizeM), g1)
B_g += pde.projections.assem_H1_b(MESH, BASIS, LISTS, dict(space = 'P1', order = 2, edges = right_block, size = sizeM), g2)

MAT = pde.assemble.hdiv(MESH, BASIS, LISTS, space = 'BDM1-BDM1');
D = MAT['BDM1-BDM1']['D'];

# A = Kxx + Kyy + 10**10*B_walls
# b = 10**10*B_g
# u = sps.linalg.spsolve(A,b)


# f = lambda x,y : 1/2*(x**2+y**2+x*y)

# c = 1;

# fx = lambda x,y : c*(x+1/2*y)
# fy = lambda x,y : c*(y+1/2*x)

# fxx = lambda x,y : c+0*x
# fxy = lambda x,y : c/2+0*x
# fyx = lambda x,y : c/2+0*x
# fyy = lambda x,y : c+0*x


fx = lambda x,y : 2*x+y+0.4*x**3
fy = lambda x,y : 2*y+x+0.4*y**3

fxx = lambda x,y : 2+1.2*x**4
fxy = lambda x,y : 1+0*x
fyx = lambda x,y : 1+0*x
fyy = lambda x,y : 2+1.2*y**4

# pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P2', matrix = 'K'))[3]

penalty = 10**10

def update_left(ux,uy):
    
    fxx_grad_u_Kxx = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'K', coeff_const = fxx(ux,uy)))[0]
    fyy_grad_u_Kyy = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'K', coeff_const = fyy(ux,uy)))[1]
    fxy_grad_u_Kxy = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'K', coeff_const = fxy(ux,uy)))[2]
    fyx_grad_u_Kyx = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'K', coeff_const = fyx(ux,uy)))[3]
    
    return (fxx_grad_u_Kxx + fyy_grad_u_Kyy + fxy_grad_u_Kxy + fyx_grad_u_Kyx)

def update_right(u,ux,uy):
    
    return -Cx.T*fx(ux,uy)-Cy.T*fy(ux,uy) -penalty*B_walls*u +penalty*B_g


def F(u,ux,uy):
    return update_right(u,ux,uy)


def WulffePowell(F,dF): # Finds proper line search parameter 
    mu = 0.01; sigma = 0.9; tau = 0.1; tau1 = 0.1; tau2 = 0.6; zeta1 = 1; zeta2 = 10; alpha = 1;
    phi_min = 0
    # 0 < mu < 1/2
    # mu < sigma < 1
    # 0 < tau < 1/2
    # 0 < tau1 < tau2 < 1
    # 1 \le zeta_1 \le zeta_2
    # alpha0>0
    alphaL = 0
    phiL = F(0)
    dphiL = dF(0)
    flag = 1
    
    for i in range(100):
        print(alphaL)
        phi_hat = F(alpha)
        if phi_hat < phi_min:
            return alpha
        if phi_hat > F(0) + mu*alpha*dF(0):
            flag = 0
            alphaR = alpha
            delta = alphaR-alphaL
            c = (F(alpha) - F(alphaL) -dF(alphaL)*delta)*1/(delta**2)
            alpha_welle = alphaL-dF(alphaL)/(2*c)
            alpha = min(max(alphaL + tau*delta,alpha_welle),alphaR-tau*delta)
        else:
            dphi_hat = dF(alpha)
            if dphi_hat < sigma*dF(0):
                if flag == 1:
                    if dphiL/dphi_hat > (1 + zeta2)/zeta2:
                        alpha_welle = alpha + (alpha-alphaL)*max(dphi_hat/(dphiL-dphi_hat),zeta1)
                    else:
                        alpha_welle = alpha + zeta2*(alpha-alphaL)
                else:
                    if dphiL/dphi_hat > 1 + (alpha-alphaL)/(tau2*(alphaR-alpha)):
                        alpha_welle = alpha + max((alpha-alphaL)*dphi_hat/(dphiL-dphi_hat),tau1*(alphaR-alpha))
                    else:
                        alpha_welle = alpha + tau2*(alphaR-alpha)
                        
                alphaL = alpha; phiL = phi_hat; dphiL = dphi_hat; alpha = alpha_welle;
            else:
                return alpha
            

# alpha = WulffePowell()

u = 1+np.zeros(shape = M.shape[0])

for i in range(100):
    ux = sps.linalg.spsolve(D,Cx*u)
    uy = sps.linalg.spsolve(D,Cy*u)
    
    Au = update_left(ux,uy) + penalty*B_walls
    rhs = update_right(u,ux,uy)
    
    w = sps.linalg.spsolve(Au,rhs)
    # u_new = u + w
    
    u_try = u
    
    for i in range(10):
        alphak = 2**(-i)
        u_try = u + alphak*w
        print(alphak)
        ux_try = sps.linalg.spsolve(D,Cx*u)
        uy_try = sps.linalg.spsolve(D,Cy*u)
        if np.linalg.norm(F(u,ux,uy))>np.linalg.norm(F(u_try,ux_try,uy_try)):
            break
    
    u = u_try
    
    # print('Norm' + str(np.linalg.norm(F(u,ux,uy))))
    if np.linalg.norm(F(u,ux,uy))<1e-5:
        break
    
    # u = u_new


# r = 

# AA = pde.projections.evaluateP1_trig(MESH,dict(),lambda x,y : x*y)

fig = MESH.pdesurf_hybrid(dict(trig = 'P1', controls = 1), u)
fig.show()

# AAx = sps.linalg.spsolve(D,Cx*AA)
# AAy = sps.linalg.spsolve(D,Cy*AA)



# fig = MESH.pdesurf_hybrid(dict(trig = 'P0', controls = 1), AAx)
# fig.show()

# from matplotlib.pyplot import spy
# spy(fxy_grad_u_Kxy)