import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

import numpy as np
import pde
import scipy.sparse as sps
from sksparse.cholmod import cholesky
import time
import gmsh


# cupyx.scipy.sparse.spmatrix

import plotly.io as pio
pio.renderers.default = 'browser'
np.set_printoptions(precision = 8)

# @profile
# def do():
    
################################################################################
# dt = 0.00125/2
T = 1.9125
dt = 0.12
iterations = 8
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
sigma_outside = lambda x,y : 0*x+0*y+0
################################################################################



################################################################################
gmsh.initialize()
gmsh.open('twoDomains.geo')
gmsh.option.setNumber("Mesh.Algorithm", 1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", init_ref)
gmsh.option.setNumber("Mesh.SaveAll", init_ref)
# gmsh.fltk.run()
p,e,t,q = pde.petq_generate()
gmsh.write("twoDomains.m")
gmsh.finalize()

MESH = pde.mesh(p,e,t,q)
################################################################################

error = np.zeros(iterations)
uh_x_P1d_fine = 0
uh_y_P1d_fine = 0

for i in range(iterations):
    print('Iteration',i+1,'out of',iterations)
    h_approx = 2/np.sqrt(MESH.nt/2)
    print('h approx',h_approx,'dt chosen exactly as ',dt, 'konstante etwa:',dt/h_approx)

    ################################################################################
    qMhx,qMhy = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'M', order = 1)
    qMx,qMy = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'M', order = 2)
    qMb2 = pde.hdiv.assembleB(MESH, space = 'BDM1', matrix = 'M', order = 2, shape = 2*MESH.NoEdges)
    
    qK = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'K', order = 2)
    qD1 = pde.l2.assemble(MESH, space = 'P1d', matrix = 'M', order = 2)
    # qD0 = pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = 1)
    
    D2 = pde.int.assemble(MESH, order = 2)
    D1 = pde.int.assemble(MESH, order = 1)
    # D0 = pde.int.assemble(MESH, order = 0)
    
    D2b = pde.int.assembleB(MESH, order = 2)
    # D1b = pde.int.assembleB(MESH, order = 1)
    # D0b = pde.int.assembleB(MESH, order = 0)
    
    sigma_outside_eval1 = pde.int.evaluate(MESH, order = 1, coeff = sigma_outside, regions = np.r_[1])
    sigma_circle_eval1  = pde.int.evaluate(MESH, order = 1, coeff = sigma_circle, regions = np.r_[2])
    
    # sigma_outside_eval2 = pde.int.evaluate(MESH, order = 2, coeff = sigma_outside, regions = np.r_[1])
    # sigma_circle_eval2  = pde.int.evaluate(MESH, order = 2, coeff = sigma_circle, regions = np.r_[2])
    
    # M = qMx@D2@(sigma_outside_eval2 + sigma_circle_eval2)@qMx.T +\
    #     qMy@D2@(sigma_outside_eval2 + sigma_circle_eval2)@qMy.T
    
    Mh_sigma = qMhx@D1@(sigma_outside_eval1 + sigma_circle_eval1)@qMhx.T +\
               qMhy@D1@(sigma_outside_eval1 + sigma_circle_eval1)@qMhy.T
    
    # epsilon = 1
    Mh_epsilon = qMhx@D1@qMhx.T +\
                 qMhy@D1@qMhy.T
    
    Mh = Mh_epsilon + dt/2*Mh_sigma
    
    K = qK@D2@qK.T
    # C = qD@D1@qK.T
    D1 = qD1@D2@qD1.T
    
    iMh = pde.tools.fastBlockInverse(Mh)
    # print(sps.linalg.norm(Mh@iMh,np.inf))
    
    iMh_Mh_sigma = iMh@Mh_sigma
    iMh_K = iMh@K
    qMb2_D2b = qMb2@D2b
    
    print('MegaBytes of iMh_K:',iMh_K.data.nbytes/(1024*1024),\
          'MegaBytes of iMh:',iMh.data.nbytes/(1024*1024),\
          'MegaBytes of iMh_Mh_sigma:',iMh_Mh_sigma.data.nbytes/(1024*1024))
    
    uh_NC1_oldold = pde.hdiv.interp(MESH, space = 'BDM1', order = 5, f = lambda x,y : np.c_[u1ex(x,y,0),u2ex(x,y,0)])
    uh_NC1_old = pde.hdiv.interp(MESH, space = 'BDM1', order = 5, f = lambda x,y : np.c_[u1ex(x,y,dt),u2ex(x,y,dt)])
    ################################################################################
    
    
    
    ################################################################################
    tm = time.monotonic()
    for j in range(int(T/dt)):
        
        jdt = j*dt
        
        intF = qMb2_D2b@ pde.int.evaluateB(MESH, order = 2, coeff = lambda x,y : divuex(x,y,jdt), edges = np.r_[1,2,3,4], like = 1)
        
        uh_NC1 = 2*uh_NC1_old-uh_NC1_oldold-(dt**2)*(iMh_K@uh_NC1_old +iMh@intF +iMh_Mh_sigma@(uh_NC1_old-uh_NC1_oldold)/dt)
        
        uh_NC1_oldold = uh_NC1_old
        uh_NC1_old = uh_NC1
        
        if (j*100//int(T/dt))%10 == 0:
            print("\rTimestepping : ",j*100//int(T/dt),'%', end = " ")
    
    print('Time stepping took a total of {:4.8f} seconds.'.format(time.monotonic()-tm))
    print('\n')
    ################################################################################
            
    uh_x_P1d = qMhx.T@uh_NC1
    uh_y_P1d = qMhy.T@uh_NC1
    
    if i>0:
        error[i] = np.sqrt((uh_x_P1d-uh_x_P1d_fine)@D1@(uh_x_P1d-uh_x_P1d_fine)+\
                           (uh_y_P1d-uh_y_P1d_fine)@D1@(uh_y_P1d-uh_y_P1d_fine))
    
    uh_x_P1d_fine = MESH.refineP1d(uh_x_P1d)
    uh_y_P1d_fine = MESH.refineP1d(uh_y_P1d)
    
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), np.sqrt(uh_x_P1d**2+uh_y_P1d**2))
    # fig.show()
    
    if i+1!=iterations:
        MESH.refinemesh(); dt = dt/2;
        
        ################################################################################
        # Shift points to the circle
        ################################################################################
        Indices_PointsOnCircle = np.unique(MESH.EdgesToVertices[MESH.Boundary_Edges[MESH.Boundary_Region==5],:].flatten())
        PointsOnCircle = MESH.p[Indices_PointsOnCircle,:]
        MESH.p[Indices_PointsOnCircle,:] = 0.3*1/np.sqrt(PointsOnCircle[:,0]**2+PointsOnCircle[:,1]**2)[:,None]*PointsOnCircle
        ################################################################################
    
    
rate = np.log2(error[1:-1]/error[2:])
print("Convergenge rates : ",rate)

# do()