import sys
sys.path.insert(0,'../SeminarEM/')
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import geometries
from sksparse.cholmod import cholesky

import plotly.io as pio
pio.renderers.default = 'browser'
from matplotlib.pyplot import spy


# np.set_printoptions(threshold = np.inf)
# np.set_printoptions(linewidth = np.inf)
# np.set_printoptions(precision = 2)

gmsh.initialize()
gmsh.model.add("Capacitor plates")
geometries.unitSquare()
gmsh.option.setNumber("Mesh.Algorithm", 2)
gmsh.option.setNumber("Mesh.MeshSizeMax", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", 1)
# gmsh.fltk.run()

########################################################

import netgen.occ as occ


########################################################

f = lambda x,y : (np.pi**2+np.pi**2)*np.sin(np.pi*x)*np.sin(np.pi*y)
g = lambda x,y : 1+0*x
u = lambda x,y : 1+np.sin(np.pi*x)*np.sin(np.pi*y)

p,e,t,q = pde.petq_generate()
MESH = pde.mesh(p,e,t,q)

iterations = 1

err = np.zeros(shape = (iterations,1))

for i in range(iterations):
    
    tm = time.time()
    
    # Mass & Stifness
    Kx,Ky = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 0)
    D0 = pde.int.assemble(MESH, order = 0)
    
    MB = pde.h1.assemble(MESH, space = 'P1', matrix = 'M', order = 2)
    D2 = pde.int.assemble(MESH, order = 2)
    
    Kxx = Kx@D0@Kx.T; Kyy = Ky@D0@Ky.T; M = MB@D2@MB.T
    
    ff = pde.int.evaluate(MESH, coeff = f, order = 2)
    M_f = MB@D2@ ff.diagonal()
    
    # Boundary stuff
    Mb = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = Kxx.shape, order = 2)
    D0b = pde.int.assembleB(MESH, order = 2)
    B_full = Mb@D0b@Mb.T
    
    
    # E1,E2,E3 = pde.h1.assembleE(MESH, space = 'P1', matrix = 'M', order = 2)
    # D0bE = pde.int.assembleE(MESH, order = 2)    
    # E = pde.tools.spxoradd(E1,E2,E3)    
    # B_fullE = E@D0bE@E.T
    
    # D_gE = pde.int.evaluateE(MESH, order = 2, coeff = g)
    
    # TODO : This should be fixed somehow... mb for H1 it doesnt really make sense to integrate over all edges
    # B_gE = np.abs(E)@D0bE@D_gE.diagonal()
    
    D_g = pde.int.evaluateB(MESH, order = 2, coeff = g)
    D_g0 = pde.int.evaluateB(MESH, order = 0, coeff = g)
    B_g = Mb@D0b@D_g.diagonal()
    
    gamma = 10**10
    
    R1,R2 = pde.h1.assembleR(MESH, space = 'P1', edges = np.r_[1,4])
    
    g_ex = g(MESH.p[:,0],MESH.p[:,1])
    
    A = Kxx + Kyy + gamma*B_full
    b = gamma*B_g + M_f
    
    new_A = R2@(Kxx+Kyy)@R2.T
    new_b = R2@M_f -R2@(Kxx+Kyy)@R1.T@(R1@g_ex)
    
    
    u_ex = u(MESH.p[:,0],MESH.p[:,1])
    elapsed = time.time()-tm; print('Assembling stuff took  {:4.8f} seconds.'.format(elapsed))
    
    
    tm = time.time()    
    # chol_A = cholesky(A); uh = chol_A(b)
    
    chol_new_A = cholesky(new_A); new_uh = chol_new_A(new_b)
    uh = R2.T@new_uh + R1.T@(R1@g_ex)
    
    elapsed = time.time()-tm; print('Solving took  {:4.8f} seconds.'.format(elapsed))
    
    err[i] = np.sqrt((u_ex-uh)@(M + Kxx + Kyy)@(u_ex-uh))
    
    if i!=iterations-1:
        tm = time.time()
        MESH.refinemesh()
        elapsed = time.time()-tm; print('Refining mesh took {:4.8f} seconds.\n'.format(elapsed))
    
    # tm = time.time()
    # MESH = pde.mesh(p,e,t,q)
    # elapsed = time.time()-tm; print('Making mesh took {:4.8f} seconds.'.format(elapsed))
    
    # tm = time.time()
    # MESH.makeFemLists(space = 'P1')
    # elapsed = time.time()-tm; print('Making lists took {:4.8f} seconds.'.format(elapsed))
    
print(np.log2(err[1:-1]/err[2:]))

# fig = MESH.pdesurf_hybrid(dict(trig = 'P1', controls = 1), u_ex)
# fig.show()

