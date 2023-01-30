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
import MaterialLaws

import plotly.io as pio
pio.renderers.default = 'browser'

np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = np.inf)
np.set_printoptions(precision = 8)

gmsh.initialize()
gmsh.model.add("Capacitor plates")
geometries.geometryP2()
gmsh.option.setNumber("Mesh.Algorithm", 2)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.MeshSizeMax",0.6)

# gmsh.fltk.run()
# quit()

p,e,t,q = pde.petq_generate()
gmsh.clear()
gmsh.finalize()

MESH = pde.mesh(p,e,t,q)
MESH.makeFemLists()

BASIS = pde.basis()
LISTS = pde.lists(MESH)

f1 = lambda x,y : -1+0*x
f2 = lambda x,y :  1+0*x

nu1 = lambda x,y : 1/1000 + 0*x +0*y
nu2 = lambda x,y : 1 + 0*x +0*y

BKx,BKy = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 0)
D2 = pde.int.assemble(MESH, order = 2)
D0 = pde.int.assemble(MESH, order = 0)

Co1 = pde.int.evaluate(MESH, order = 0, coeff = nu1, regions = np.r_[2,3])
Co2 = pde.int.evaluate(MESH, order = 0, coeff = nu2, regions = np.r_[1,4,5,6,7,8])

Kxx = BKx@D0@(Co1+Co2)@BKx.T
Kyy = BKy@D0@(Co1+Co2)@BKy.T

nu_aus = Co1.diagonal()+Co2.diagonal()

BM = pde.h1.assemble(MESH, space = 'P1', matrix = 'M', order = 2)
M = BM@D2@BM.T

D = pde.l2.assemble(MESH, space = 'P0', matrix = 'M')

CoF1 = pde.int.evaluate(MESH, order = 2, coeff = f1, regions = np.r_[7])
CoF2 = pde.int.evaluate(MESH, order = 2, coeff = f2, regions = np.r_[8])


M_f = BM@D2@(CoF1.diagonal()+CoF2.diagonal())

Mb = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = Kxx.shape)
Db2 = pde.int.assembleB(MESH, order = 2)

B = Mb@Db2@Mb.T

BKx,BKy = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 0)
BD = pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = 0)
Cx = BD @ D0 @ BKx.T
Cy = BD @ D0 @ BKy.T

penalty = 10**10
A = Kxx + Kyy + penalty*B
b = M_f


# fx = lambda x,y : 2*x+y+0.4*x**3
# fy = lambda x,y : 2*y+x+0.4*y**3

# fxx = lambda x,y : 2+1.2*x**4
# fxy = lambda x,y : 1+0*x
# fyx = lambda x,y : 1+0*x
# fyy = lambda x,y : 2+1.2*y**4

fx = lambda x,y : x
fy = lambda x,y : y

fxx = lambda x,y : 1+0*x
fxy = lambda x,y : 0*x
fyx = lambda x,y : 0*x
fyy = lambda x,y : 1+0*y

g,dg,ddg = MaterialLaws.HerbertsMaterialG(a = 0.1, b = 1)


penalty = 10**7

def update_left(ux,uy):
    
    fxx_grad_u_Kxx = BKx @ D0 @ sps.diags(ddg(ux,uy,nu_aus)[0,0,:])@ BKx.T
    fyy_grad_u_Kyy = BKy @ D0 @ sps.diags(ddg(ux,uy,nu_aus)[1,1,:])@ BKy.T
    fxy_grad_u_Kxy = BKy @ D0 @ sps.diags(ddg(ux,uy,nu_aus)[1,0,:])@ BKx.T
    fyx_grad_u_Kyx = BKx @ D0 @ sps.diags(ddg(ux,uy,nu_aus)[0,1,:])@ BKy.T
    
    return (fxx_grad_u_Kxx + fyy_grad_u_Kyy + fxy_grad_u_Kxy + fyx_grad_u_Kyx)

def update_right(u,ux,uy):
    
    return -Cx.T @ dg(ux,uy,nu_aus)[0,:] -Cy.T @ dg(ux,uy,nu_aus)[1,:] + M_f -penalty*B@u



u = 1+np.zeros(shape = Kxx.shape[0])

for i in range(40):
    
    ux = BKx.T@u
    uy = BKy.T@u
    
    Au = update_left(ux,uy) + penalty*B
    rhs = update_right(u,ux,uy)
    
    w = sps.linalg.spsolve(Au,rhs)
    u_new = u + w
    
    if np.linalg.norm(w)<1e-16:
        break
    print(np.linalg.norm(w))
    u = u_new




# tm = time.time()
# u = sps.linalg.spsolve(A,b)
# elapsed = time.time()-tm
# print('Solving took ' + str(elapsed)[0:5] + ' seconds.')

# ux = BKx.T@u
# uy = BKy.T@u

fig = MESH.pdesurf_hybrid(dict(trig = 'P1',quad = 'Q1', controls = 1), u)
fig.show()