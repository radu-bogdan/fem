import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../CEM')

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import geometries

import plotly.io as pio
pio.renderers.default = 'browser'
from matplotlib.pyplot import spy
from sksparse.cholmod import cholesky


np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = np.inf)
np.set_printoptions(precision = 8)

gmsh.initialize()
gmsh.model.add("Capacitor plates")
geometries.unitSquare()
gmsh.option.setNumber("Mesh.Algorithm", 1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", 1)
# gmsh.fltk.run()

m = 4; n = 5;

f = lambda x,y : (m**2*np.pi**2+n**2*np.pi**2)*np.sin(np.pi*m*x)*np.sin(np.pi*n*y)
g = lambda x,y : 0*x+0*y
u = lambda x,y : np.sin(np.pi*m*x)*np.sin(np.pi*n*y)

p,e,t,q = pde.petq_generate()
MESH = pde.mesh(p,e,t,q)
MESH.makeFemLists(space = 'P2')

iterations = 10

err = np.empty(shape = (iterations,1))
for i in range(iterations):
    print(i)
    tm = time.time()
    
    # Mass & Stifness    
    Kx,Ky = pde.h1.assemble(MESH, space = 'P2', matrix = 'K', order = 2)
    D2 = pde.int.assemble(MESH, order = 2)
    
    MB = pde.h1.assemble(MESH, space = 'P2', matrix = 'M', order = 4)
    D4 = pde.int.assemble(MESH, order = 4)
    
    Kxx = Kx@D2@Kx.T; Kyy = Ky@D2@Ky.T; M = MB@D4@MB.T
    
    ff = pde.int.evaluate(MESH, coeff = f, order = 4)
    M_f = MB@D4@ ff.diagonal()
    
    # Boundary stuff
    Mb = pde.h1.assembleB(MESH, space = 'P2', matrix = 'M', shape = Kxx.shape, order = 5)
    Db0 = pde.int.assembleB(MESH, order = 5)
    B_full = Mb@Db0@Mb.T
    
    D_g = pde.int.evaluateB(MESH, order = 5, coeff = g)
    B_g = Mb@Db0@ D_g.diagonal()
    
    gamma = 10**13
    
    A = Kxx + Kyy + gamma*B_full
    b = gamma*B_g + M_f
    
    
    mp = 1/2*(MESH.p[MESH.EdgesToVertices[:,0],:] + 
              MESH.p[MESH.EdgesToVertices[:,1],:])
    
    u_ex = np.r_[u(MESH.p[:,0],MESH.p[:,1]),
                 u(mp[:,0],mp[:,1])]
    
    elapsed = time.time()-tm; print('Assembling stuff took {:4.8f} seconds.'.format(elapsed))
    
    # u = MB@D2@ pde.int.evaluate(MESH, coeff = f, order = 2).diagonal()
    
    # sigma = 3
    # # x = sps.linalg.eigs(Kxx+Kyy-sigma*M+gamma*B_full,M = M, sigma = sigma)
    # x = sps.linalg.eigs(Kxx+Kyy,M = M, sigma = sigma)
    # phi = np.real(x[1][:,0])
    
    tm = time.time()
    factor = cholesky(A)
    uh = factor(b)
    elapsed = time.time()-tm
    print('Solving took {:4.8f} seconds.'.format(elapsed))
    
    err[i] = np.sqrt((u_ex-uh)@M@(u_ex-uh)+0*(u_ex-uh)@(Kxx+Kyy)@(u_ex-uh))
    # err[i] = np.sqrt((u_ex-uh)@M@(u_ex-uh)+(u_ex-uh)@(Kxx+Kyy)@(u_ex-uh))
    
    if i!=iterations-1:
        tm = time.time()
        MESH.refinemesh()
        elapsed = time.time()-tm; print('Refining mesh took {:4.8f} seconds.'.format(elapsed))
        
        tm = time.time()
        MESH.makeFemLists(space = 'P2')
        elapsed = time.time()-tm; print('Making lists took {:4.8f} seconds.'.format(elapsed))
        
        print(MESH.np)
    
print(np.log2(err[1:-1]/err[2:]))

# fig = MESH.pdesurf_hybrid(dict(trig = 'P1', controls = 1), u_ex[0:MESH.np])
# fig.show()

