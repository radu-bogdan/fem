import sys
sys.path.insert(0,'../../') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import geometries

import plotly.io as pio
pio.renderers.default = 'browser'


##########################################################################################
# Loading mesh
##########################################################################################
motor_npz = np.load('meshes/motor.npz', allow_pickle=True)

p = motor_npz['p'].T
e = motor_npz['e'].T
t = motor_npz['t'].T
q = np.empty(0)
regions_2d = motor_npz['regions_2d']
regions_1d = motor_npz['regions_1d']
m = motor_npz['m']
j3 = motor_npz['j3']

MESH = pde.mesh(p,e,t,q)
##########################################################################################



##########################################################################################
# Extract indices
##########################################################################################
ind_air_all = np.flatnonzero(np.core.defchararray.find(list(regions_2d),'air')!=-1)
ind_stator_rotor = np.flatnonzero(np.core.defchararray.find(list(regions_2d),'iron')!=-1)
ind_magnet = np.flatnonzero(np.core.defchararray.find(list(regions_2d),'magnet')!=-1)
ind_coil = np.flatnonzero(np.core.defchararray.find(list(regions_2d),'coil')!=-1)

ind_shaft = np.flatnonzero(np.core.defchararray.find(list(regions_2d),'shaft')!=-1)

trig_air_all = np.where(np.isin(t[:,3],ind_air_all))[0]
trig_stator_rotor_and_shaft = np.where(np.isin(t[:,3],ind_stator_rotor))[0]
trig_magnet = np.where(np.isin(t[:,3],ind_magnet))[0]
trig_coil = np.where(np.isin(t[:,3],ind_coil))[0]
trig_shaft = np.where(np.isin(t[:,3],ind_shaft))[0]
trig_stator_rotor = np.setdiff1d(trig_stator_rotor_and_shaft,trig_shaft)
##########################################################################################




##########################################################################################
# Brauer/Nonlinear laws ... ?
##########################################################################################
k1 = 49.4; k2 = 1.46; k3 = 520.6

f_iron = lambda x,y : k1/(2*k2)*(np.exp(k2*(x**2+y**2))-1) + 1/2*k3*(x**2+y**2) # magnetic energy density in iron
nu = lambda x,y : k1*np.exp(k2*(x**2+y**2))+k3
nux = lambda x,y : 2*x*k1*k2*np.exp(k2*(x**2+y**2))
nuy = lambda x,y : 2*y*k1*k2*np.exp(k2*(x**2+y**2))
fx_iron = lambda x,y : nu(x,y)*x
fy_iron = lambda x,y : nu(x,y)*y
fxx_iron = lambda x,y : nu(x,y) + x*nux(x,y)
fxy_iron = lambda x,y : x*nuy(x,y)
fyx_iron = lambda x,y : y*nux(x,y)
fyy_iron = lambda x,y : nu(x,y) + y*nuy(x,y)

nu0 = 10**7/(4*np.pi)

f_linear = lambda x,y : 1/2*nu0*(x**2+y**2)
fx_linear = lambda x,y : nu0*x
fy_linear = lambda x,y : nu0*y
fxx_linear = lambda x,y : nu0(x,y)
fxy_linear = lambda x,y : x*0
fyx_linear = lambda x,y : y*0
fyy_linear = lambda x,y : nu0(x,y)

def f(ux,uy):
    f_linear_eval  = f_linear(ux,uy); f_linear_eval[trig_stator_rotor] = 0  
    f_iron_eval = f_iron(ux,uy); f_iron_eval[np.r_[trig_air_all,trig_magnet,trig_shaft]] = 0
    return f_linear_eval + f_iron_eval

def fx(ux,uy):
    fx_linear_eval  = fx_linear(ux,uy); fx_linear_eval[trig_stator_rotor] = 0  
    fx_iron_eval = fx_iron(ux,uy); fx_iron_eval[np.r_[trig_air_all,trig_magnet,trig_shaft]] = 0
    return fx_linear_eval + fx_iron_eval

def fy(ux,uy):
    fy_linear_eval  = fy_linear(ux,uy); fy_linear_eval[trig_stator_rotor] = 0  
    fy_iron_eval = fy_iron(ux,uy); fy_iron_eval[np.r_[trig_air_all,trig_magnet,trig_shaft]] = 0
    return fy_linear_eval + fy_iron_eval

def fxx(ux,uy):
    fxx_linear_eval  = fxx_linear(ux,uy); fxx_linear_eval[trig_stator_rotor] = 0  
    fxx_iron_eval = fxx_iron(ux,uy); fxx_iron_eval[np.r_[trig_air_all,trig_magnet,trig_shaft]] = 0
    return fxx_linear_eval + fxx_iron_eval

def fyy(ux,uy):
    fyy_linear_eval  = fyy_linear(ux,uy); fyy_linear_eval[trig_stator_rotor] = 0  
    fyy_iron_eval = fyy_iron(ux,uy); fyy_iron_eval[np.r_[trig_air_all,trig_magnet,trig_shaft]] = 0
    return fyy_linear_eval + fyy_iron_eval

def fxy(ux,uy):
    fxy_linear_eval  = fxy_linear(ux,uy); fxy_linear_eval[trig_stator_rotor] = 0  
    fxy_iron_eval = fxy_iron(ux,uy); fxy_iron_eval[np.r_[trig_air_all,trig_magnet,trig_shaft]] = 0
    return fxy_linear_eval + fxy_iron_eval

def fyx(ux,uy):
    fyx_linear_eval  = fyx_linear(ux,uy); fyx_linear_eval[trig_stator_rotor] = 0  
    fyx_iron_eval = fyx_iron(ux,uy); fyx_iron_eval[np.r_[trig_air_all,trig_magnet,trig_shaft]] = 0
    return fyx_linear_eval + fyx_iron_eval
##########################################################################################

dphix_H1,dphiy_H1 = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 0)
D0 = pde.int.assemble(MESH, order = 0)
D1 = pde.int.assemble(MESH, order = 1)

Kxx = dphix_H1 @ D0 @ dphix_H1.T; Kyy = dphiy_H1 @ D0 @ dphiy_H1.T

phi_L2 = pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = 0)
Cx = phi_L2 @ D0 @ dphix_H1.T
Cy = phi_L2 @ D0 @ dphiy_H1.T


def update_left(ux,uy):
    fxx_grad_u_Kxx = dphix_H1 @ D0 @ sps.diags(fxx(ux,uy))@ dphix_H1.T
    fyy_grad_u_Kyy = dphiy_H1 @ D0 @ sps.diags(fyy(ux,uy))@ dphiy_H1.T
    fxy_grad_u_Kxy = dphiy_H1 @ D0 @ sps.diags(fxy(ux,uy))@ dphix_H1.T
    fyx_grad_u_Kyx = dphix_H1 @ D0 @ sps.diags(fyx(ux,uy))@ dphiy_H1.T



def update_right(u,ux,uy):
    
    return -Cx.T @ fx(ux,uy) -Cy.T @ fy(ux,uy) -penalty*B_walls@u + penalty*B_g


