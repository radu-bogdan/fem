import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
from sksparse.cholmod import cholesky
import time
import gmsh
import geometries

import plotly.io as pio
pio.renderers.default = 'browser'
np.set_printoptions(precision = 8)


gmsh.initialize()
gmsh.open('twoDomains.geo')
gmsh.option.setNumber("Mesh.Algorithm", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.01)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
# gmsh.fltk.run()
p,e,t,q = pde.petq_generate()
gmsh.finalize()


MESH = pde.mesh(p,e,t,q)
# MESH.refinemesh()
# MESH.refinemesh()
# MESH.refinemesh()
# MESH.refinemesh()
# MESH.refinemesh()

# fig = MESH.pdemesh()
# fig.show()

################################################################################
dt = 0.00125/2
T = 2.2
time = 3060
iteration = 1
init_ref = 1


kx = 1; ky = 1; s0 = -3
c = np.sqrt(kx**2+ky**2)

g = lambda s : 2*np.exp(-10*(s-s0)**2)
gs = lambda s : 2*np.exp(-10*(s-s0)**2)*(-20*(s-s0))

pex = lambda x,y,t : g(kx*x+ky*y-c*t)
u1ex = lambda x,y,t : kx/c*pex(x,y,t)
u2ex = lambda x,y,t : ky/c*pex(x,y,t)
divuex = lambda x,y,t : (kx**2+ky**2)/c*gs(kx*x+ky*y-c*t)


sigma_circle = lambda x,y : 0*x+0*y+100
sigma_outside = lambda x,y : 0*x+0*y+1
################################################################################



################################################################################
qMhx,qMhy = pde.hcurl.assemble(MESH, space = 'NC1', matrix = 'M', order = 1)
qMx,qMy = pde.hcurl.assemble(MESH, space = 'NC1', matrix = 'M', order = 2)
qMb2 = pde.hcurl.assembleB(MESH, space = 'NC1', matrix = 'M', order = 2, shape = 2*MESH.NoEdges)

qK = pde.hcurl.assemble(MESH, space = 'NC1', matrix = 'K', order = 2)
qD = pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = 1)

D2 = pde.int.assemble(MESH, order = 2)
D1 = pde.int.assemble(MESH, order = 1)
D0 = pde.int.assemble(MESH, order = 0)

D2b = pde.int.assembleB(MESH, order = 2)
D1b = pde.int.assembleB(MESH, order = 1)
D0b = pde.int.assembleB(MESH, order = 0)

sigma_outside_eval1 = pde.int.evaluate(MESH, order = 1, coeff = sigma_circle, regions = np.r_[1])
sigma_circle_eval1  = pde.int.evaluate(MESH, order = 1, coeff = sigma_circle, regions = np.r_[2])

sigma_outside_eval2 = pde.int.evaluate(MESH, order = 2, coeff = sigma_circle, regions = np.r_[1])
sigma_circle_eval2  = pde.int.evaluate(MESH, order = 2, coeff = sigma_circle, regions = np.r_[2])

M = qMx@D2@(sigma_outside_eval2 + sigma_circle_eval2)@qMx.T +\
    qMy@D2@(sigma_outside_eval2 + sigma_circle_eval2)@qMy.T

Mh_sigma = qMhx@D1@(sigma_outside_eval1 + sigma_circle_eval1)@qMhx.T +\
           qMhy@D1@(sigma_outside_eval1 + sigma_circle_eval1)@qMhy.T

# epsilon = 1
Mh_epsilon = qMhx@D1@qMhx.T +\
             qMhy@D1@qMhy.T

Mh = Mh_epsilon + dt/2*Mh_sigma

K = qK@D2@qK.T
C = qD@D1@qK.T
D = qD@D1@qD.T

iMh = pde.tools.fastBlockInverse(Mh)
print(sps.linalg.norm(Mh@iMh,np.inf))

iMh_Mh_sigma = iMh@Mh_sigma
iMh_K = iMh@K

# uh_NC1 = pde.hcurl.interp(MESH, space = 'NC1', order = 5, f = lambda x,y : np.c_[u1ex(x,y,2),u2ex(x,y,2)])

# # uh_P1d = qMhx.T@uh_NC1
# # uh_P1d2 = pde.int.evaluate(MESH, order = 1, coeff = lambda x,y : u1ex(x,y,2), regions = np.r_[1,2]).diagonal()
# # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), uh_P1d-uh_P1d2)
# # fig.show()

uh_NC1_oldold = pde.hcurl.interp(MESH, space = 'NC1', order = 5, f = lambda x,y : np.c_[u1ex(x,y,0),u2ex(x,y,0)])
uh_NC1_old = pde.hcurl.interp(MESH, space = 'NC1', order = 5, f = lambda x,y : np.c_[u1ex(x,y,dt),u2ex(x,y,dt)])
################################################################################


z = 0
################################################################################
for j in range(time):
    
    jdt = j*dt
    
    fb2 = qMb2@D2b@ pde.int.evaluateB(MESH, order = 2, coeff = lambda x,y : divuex(x,y,jdt), edges = np.r_[1,2,3,4]).diagonal()
    
    uh_NC1 = 2*uh_NC1_old-uh_NC1_oldold-dt**2*(iMh_K@uh_NC1_old -iMh@fb2 +iMh_Mh_sigma@(uh_NC1_old-uh_NC1_oldold)/dt)
    
    uh_NC1_oldold = uh_NC1_old
    uh_NC1_old = uh_NC1
    
    if j>z*time/10:
        z+=1
        print(z, end = " ")
    
    
    # print(np.where(fb2>1e-12)[0].shape)
    
    # if j%200==0:
        # uh_P1d = qMhx.T@uh_NC1
        # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), uh_P1d)
        # fig.show()

################################################################################

uh_P1d = qMhx.T@uh_NC1
fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), uh_P1d)
fig.show()




from matplotlib.pyplot import spy
# spy(Mh, markersize=1)
cholMh = cholesky(Mh)
spy(cholMh.L(), markersize=1)
