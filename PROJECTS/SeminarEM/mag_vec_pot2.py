#!/usr/bin/env python
# coding: utf-8

# # Electrical machines seminar, first exercise
# ## Solving the magnetic vector potential formulation
# 
# The goal for the first project is to compute the magnetostatic problem for a "real" world problem.
# 
# To this end, we will work with Alessio \& Peter's motor geometry of a permanent magnet machine.
# 
# 
# 
# For now, let $\Omega$ denote the computational domain.
# 
# Then Maxwell's equations for magnetostatics are described by:
# 
# \begin{alignat}{2}
# \operatorname{curl} h &= j, \qquad && \text{in } \Omega, \label{eq:1}\\
# \operatorname{div} b &= 0, \qquad && \text{in } \Omega, \label{eq:2}\\
# n \cdot b &= 0, \qquad && \text{on } \partial\Omega, \label{eq:3}
# \end{alignat}
# 
# On simply connected domains, the second equation implies the existence of a vector potential $a$ such that
# 
# \begin{align}
#     \operatorname{curl} a &= b\label{eq:4}
# \end{align}
# 
# So far, these equations are independent of the materials.
# 
# To incorporate material properties, we need to introduce constitutive equations, which will be different depending on the material type, such as air, iron, or magnet.
# 
# For now, assume that the magnetic field $h$ and magnetic flux $b$ are linked by
# 
# \begin{align}
#     h = f'(b) \label{eq:nl}
# \end{align}
# 
# where $f$ denotes a smooth and strongly convex "energy density" $f: \mathbb{R}^d \to \mathbb{R}$.
# 
# Using this relation, we can rewrite the problem in a compact manner in the following way:
# 
# \begin{align*}
#     \operatorname{curl}(f'(\operatorname{curl} a)) &= j + \operatorname{curl} m \\
#     n\cdot \operatorname{curl} a &=0
# \end{align*}
# 
# In 2D, we can make the following simplifications:
# 
# In the typical setting with cylindrical symmetry, which is of relevance for electric motors, one has $\Omega = \Omega_{2D} \times (0,L)$ and assumes that $j=(0,0,j_3)$, $m=(m_1,m_2,0)$, $a=(0,0,a_3)$, $b=(b_1,b_2,0)$, and $h=(h_1,h_2,0)$, with all components independent of $z$.
# 
# In this way, one arrives at the scalar problem 
# 
# \begin{alignat}{2}
# \operatorname{Curl} f'(\operatorname{curl}_{2D} a_3) &= j_3 + \operatorname{Curl}_{2D} m, \qquad && \text{in } \Omega_{2D}, \\
# n \cdot \operatorname{curl}_{2D} a_3 &= 0, \qquad && \text{on } \partial\Omega_{2D}.
# \end{alignat}
# 
# Here $\operatorname{Curl}_{2D}=(-\partial_y, \partial_x)$ and $\operatorname{curl}_{2D}=\binom{\partial_y}{-\partial_x}$ are the vector-to-scalar and scalar-to-vector curl, respectively.
# 
# This problem can further be transformed into 
# 
# \begin{alignat}{2}
# -\operatorname{div} (g'(\nabla u)) &= j_3 + \operatorname{div} m^\perp :=\widehat j_3, \qquad &&\text{in } \Omega_{2D} \label{eq:sys1} \\
# u &= 0, \qquad && \text{on } \partial\Omega_{2D} \label{eq:sys2}
# \end{alignat}
# 
# where now $u=a_3$ is the $z$-component of the vector potential, $\binom{-b_2}{b_1}=\nabla u$ is the rotated magnetic flux, $\binom{-h_2}{h_1}=g'(\nabla u)$ is the rotated magnetic field, and $m^\perp = \binom{-m_2}{m_1}$ is the rotated magnetization.
# 
# \medskip
# 
# ### Herbert's note during our first meeting:
# The boundary condition for the transformed problem reads $\partial_t u := t\cdot\nabla u=0$, meaning that the tangential component of $u$ vanishes on the whole boundary. This, in turn, means that $u$ has to be constant on the boundary. For this reason, we may set $u=0$ on $\partial\Omega$. Careful! If the domain is not simply connected, we are not allowed to prescribe $u$ everywhere on the boundary!

# Loading imports ...

# In[1]:


import sys; sys.path.insert(0,'../../') # adds parent directory
import time
import numpy as np
import pde
import scipy.sparse as sps
from sksparse.cholmod import cholesky as chol

ORDER = 1


# Loading the motor, generating inital mesh

# In[2]:


motor_npz = np.load('../meshes/motor.npz', allow_pickle = True)

m = motor_npz['m']
j3 = motor_npz['j3']

geoOCCmesh = motor_npz['geoOCCmesh'].tolist()
MESH = pde.mesh.netgen(geoOCCmesh)


# Setting up the function $f$ and its derivatives

# In[3]:


sys.path.insert(1,'../mixed.EM')
from nonlinLaws import *


# Setting first or second order variables, expanding mask for second order, as $f$ has to be evaluated for $P_1$, where the gradiant of our solution lies

# In[4]:


if ORDER == 1:
    poly = 'P1'
    dxpoly = 'P0'
    order_phi = 1
    order_dphi = 0
    order_phiphi = order_phi*2
    order_dphidphi = order_dphi*2
    u = np.zeros(MESH.np)
    
if ORDER == 2:
    poly = 'P2'
    dxpoly = 'P1'
    order_phi = 2
    order_dphi = 2
    order_phiphi = order_phi*2
    order_dphidphi = order_dphi*2
    u = np.zeros(MESH.np + MESH.NoEdges)


# Assembling the matrix and right-hand sides

# In[5]:


tm = time.monotonic()

linear = '*air,*magnet,*coil,shaft_iron'
nonlinear = 'stator_iron,rotor_iron'
rotor = '*magnet,rotor_iron,rotor_air,shaft_iron'
    
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


D_stator_outer = pde.int.evaluateB(MESH, order = order_phiphi, edges = 'stator_outer')
B_stator_outer = phi_H1b@ D_stator_outer @ phi_H1b.T


fem_linear = pde.int.evaluate(MESH, order = order_dphidphi, regions = linear).diagonal()
fem_nonlinear = pde.int.evaluate(MESH, order = order_dphidphi, regions = nonlinear).diagonal()
fem_rotor = pde.int.evaluate(MESH, order = order_dphidphi, regions = rotor).diagonal()
fem_air_gap_rotor = pde.int.evaluate(MESH, order = order_dphidphi, regions = 'air_gap_rotor').diagonal()

penalty = 1e10

J = 0; # J0 = 0
for i in range(48):
    J += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : j3[i], regions = 'coil'+str(i+1)).diagonal()
    # J0+= pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : j3[i], regions = np.r_[ind_trig_coils[i]]).diagonal()
J = 0*J

M0 = 0; M1 = 0; M00 = 0
for i in range(16):
    M0 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m[0,i], regions = 'magnet'+str(i+1)).diagonal()
    M1 += pde.int.evaluate(MESH, order = order_phiphi, coeff = lambda x,y : m[1,i], regions = 'magnet'+str(i+1)).diagonal()

    # M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()

aJ = phi_H1@ D_order_phiphi @J

aM = dphix_H1_order_phiphi@ D_order_phiphi @(-M1) +\
     dphiy_H1_order_phiphi@ D_order_phiphi @(+M0)

aMnew = aM


# fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1), M00, u_height=0)
# fig.show()

# print('Assembling + stuff ', time.monotonic()-tm)


# TODO

# In[ ]:


maxIter = 100
epsangle = 1e-5;

angleCondition = np.zeros(5)
eps_newton = 1e-8
factor_residual = 1/2
mu = 0.0001

def gss(u):
        ux = dphix_H1.T@u; uy = dphiy_H1.T@u
        
        fxx_grad_u_Kxx = dphix_H1 @ D_order_dphidphi @ sps.diags(fxx_linear(ux,uy)*fem_linear + fxx_nonlinear(ux,uy)*fem_nonlinear)@ dphix_H1.T
        fyy_grad_u_Kyy = dphiy_H1 @ D_order_dphidphi @ sps.diags(fyy_linear(ux,uy)*fem_linear + fyy_nonlinear(ux,uy)*fem_nonlinear)@ dphiy_H1.T
        fxy_grad_u_Kxy = dphiy_H1 @ D_order_dphidphi @ sps.diags(fxy_linear(ux,uy)*fem_linear + fxy_nonlinear(ux,uy)*fem_nonlinear)@ dphix_H1.T
        fyx_grad_u_Kyx = dphix_H1 @ D_order_dphidphi @ sps.diags(fyx_linear(ux,uy)*fem_linear + fyx_nonlinear(ux,uy)*fem_nonlinear)@ dphiy_H1.T
        return (fxx_grad_u_Kxx + fyy_grad_u_Kyy + fxy_grad_u_Kxy + fyx_grad_u_Kyx) + penalty*B_stator_outer   

def gs(u):    
        ux = dphix_H1.T@u; uy = dphiy_H1.T@u
        return dphix_H1 @ D_order_dphidphi @ (fx_linear(ux,uy)*fem_linear + fx_nonlinear(ux,uy)*fem_nonlinear) +\
               dphiy_H1 @ D_order_dphidphi @ (fy_linear(ux,uy)*fem_linear + fy_nonlinear(ux,uy)*fem_nonlinear) + penalty*B_stator_outer@u - aJ + aM
    
def J(u):
    ux = dphix_H1.T@u; uy = dphiy_H1.T@u
    return np.ones(D_order_dphidphi.size)@ D_order_dphidphi @(f_linear(ux,uy)*fem_linear + f_nonlinear(ux,uy)*fem_nonlinear) -(aJ-aM)@u + 1/2*penalty*u@B_stator_outer@u
    
tm = time.monotonic()
for i in range(maxIter):
    gsu = gs(u)
    gssu = gss(u)
    w = chol(gssu).solve_A(-gsu)
    # w = sps.linalg.spsolve(gssu,-gsu)

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
    #     if np.linalg.norm(gs(u+alpha*w)) <= np.linalg.norm(gs(u)): break
    #     else: alpha = alpha*factor_residual

    # AmijoBacktracking
    float_eps = np.finfo(float).eps
    for kk in range(1000):
        if J(u+alpha*w)-J(u) <= alpha*mu*(gsu@w) + np.abs(J(u))*float_eps: break
        else: alpha = alpha*factor_residual

    u = u + alpha*w

    print ("NEWTON: Iteration: %2d " %(i+1)+"||obj: %2e" %J(u)+"|| ||grad||: %2e" %np.linalg.norm(gs(u))+"||alpha: %2e" % (alpha))

    if(np.linalg.norm(gs(u)) < eps_newton): break

elapsed = time.monotonic()-tm


# In[ ]:


MESH.pdesurf2(u[:MESH.np])

ux = dphix_H1_o0.T@u
uy = dphiy_H1_o0.T@u
norm_ux = np.sqrt(ux**2+uy**2)
fig = MESH.pdesurf2(norm_ux)


# In[7]:


print(np.finfo(float).eps)


# In[ ]:




