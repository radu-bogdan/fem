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

from matplotlib.pyplot import spy


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
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
# gmsh.fltk.run()
p,e,t,q = pde.petq_generate()

MESH = pde.initmesh(p,e,t,q)

# TODO:  MESH = pde.refinemesh(p,e,t,q)
BASIS = pde.basis()
LISTS = pde.lists(MESH)

g1 = lambda x,y : -1+0*x
g2 = lambda x,y :  1+0*x

# Kxx, Kyy, Kxy, Kyx = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P2', matrix = 'K'))
# M = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P2', matrix = 'M'))

M = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P2', matrix = 'M'))

sizeM = M.shape[0]

walls = np.r_[5,6,7,8,9,10,11,12]
left_block = np.r_[5,6,7,8]
right_block = np.r_[9,10,11,12]

B_full = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P2', size = sizeM))
B_left_block = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P2', size = sizeM, edges = left_block))
B_right_block = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P2', size = sizeM, edges = right_block))
B_walls = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P2', size = sizeM, edges = walls))
# M_f = pde.projections.assem(MESH, BASIS, LISTS, dict(trig = 'P1'), f_rhs)
Cx,Cy = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P2', matrix = 'C'))

B_g  = pde.projections.assem_H1_b(MESH, BASIS, LISTS, dict(space = 'P2', order = 2, edges = left_block, size = sizeM), g1)
B_g += pde.projections.assem_H1_b(MESH, BASIS, LISTS, dict(space = 'P2', order = 2, edges = right_block, size = sizeM), g2)

MAT = pde.assemble.hdiv(MESH, BASIS, LISTS, space = 'RT1-BDFM1'); 
D = MAT['RT1-BDFM1']['D'];


fx = lambda x,y : 2*x+y+0.4*x**3
fy = lambda x,y : 2*y+x+0.4*y**3

fxx = lambda x,y : 2+1.2*x**4
fxy = lambda x,y : 1+0*x
fyx = lambda x,y : 1+0*x
fyy = lambda x,y : 2+1.2*y**4

penalty = 10**10

def update_left(ux,uy):
    
    fxx_grad_u_Kxx = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P2', matrix = 'K', coeff_const = fxx(ux,uy)))[0]
    fyy_grad_u_Kyy = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P2', matrix = 'K', coeff_const = fyy(ux,uy)))[1]
    fxy_grad_u_Kxy = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P2', matrix = 'K', coeff_const = fxy(ux,uy)))[2]
    fyx_grad_u_Kyx = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P2', matrix = 'K', coeff_const = fyx(ux,uy)))[3]
    
    return (fxx_grad_u_Kxx + fyy_grad_u_Kyy + fxy_grad_u_Kxy + fyx_grad_u_Kyx)

def update_right(u,ux,uy):
    
    return -Cx.T*fx(ux,uy)-Cy.T*fy(ux,uy) -penalty*B_walls*u +penalty*B_g



u = 1+np.zeros(shape = M.shape[0])

for i in range(100):
    ux = sps.linalg.spsolve(D,Cx*u)
    uy = sps.linalg.spsolve(D,Cy*u)
    
    Au = update_left(ux,uy) + penalty*B_walls
    rhs = update_right(u,ux,uy)
    
    w = sps.linalg.spsolve(Au,rhs)
    u_new = u + w
    
    if np.linalg.norm(rhs)<1e-7:
        break
    
    u = u_new


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