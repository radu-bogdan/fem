import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

# import line_profiler
# profile = line_profiler.LineProfiler()

from sparse_dot_mkl import dot_product_mkl as mult
import numpy as np
import gc
import pde
import scipy.sparse as sps
# from sksparse.cholmod import cholesky
import time
import gmsh
import reduction_matrix

# import torch
# import os
# local_rank = int(os.environ["LOCAL_RANK"])
# model = torch.nn.parallel.DistributedDataParallel(
#     model,
#     device_ids=[local_rank],
#     output_device=local_rank,
# )
# import torch.distributed as dist
# dist.init_process_group(backend="gloo")

# cupyx.scipy.sparse.spmatrix

import plotly.io as pio
pio.renderers.default = 'browser'
np.set_printoptions(precision = 8)

# @profile
# def do():
        
################################################################################
# dt = 0.00125/2
# T = 1.9125
# dt = (0.12/2/2)
# iterations = 1
# init_ref = 0.25
# use_GPU = False

T = 2
dt = 0.03/2*2*1.8
iterations = 9
init_ref = 0.25



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
gmsh.option.setNumber("Mesh.MeshSizeMax", init_ref)
gmsh.option.setNumber("Mesh.MeshSizeMin", init_ref)
gmsh.option.setNumber("Mesh.SaveAll", 1)
# gmsh.fltk.run()
p,e,t,q = pde.petq_generate()
# gmsh.write("twoDomains.m")
# gmsh.finalize()

MESH = pde.mesh(p,e,t,q)
# MESH.refinemesh()
################################################################################

error = np.zeros(iterations)
new_error = np.zeros(iterations)
error2 = np.zeros(iterations)
error3 = np.zeros(iterations)

dtau_uh_x_P1d_fine = np.empty(0); dtau_new_uh_x_P1d_fine = np.empty(0); dtau_new_uh0_x_P1d_fine = np.empty(0)
dtau_uh_y_P1d_fine = np.empty(0); dtau_new_uh_y_P1d_fine = np.empty(0); dtau_new_uh0_y_P1d_fine = np.empty(0)
mean_div_uh_NC1_P0_fine = np.empty(0); mean_new_div_uh_N0_P0_fine = np.empty(0); mean_new_div_uh_N00_P0_fine = np.empty(0)

for i in range(iterations):
    
    print('Iteration', i+1, 'out of', iterations)
    h_approx = 2/np.sqrt(MESH.nt/2)
    h_richtig = MESH.h().min()
    print('h approx',h_approx,'h richtig',h_richtig,'dt chosen exactly as ',dt, 'konstante etwa:',dt/h_richtig)

    ################################################################################
    qMhx,qMhy = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'M', order = 1)
    qMx,qMy = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'M', order = 2)
    qMb2 = pde.hdiv.assembleB(MESH, space = 'BDM1', matrix = 'M', order = 2, shape = 2*MESH.NoEdges)
    qMb2_RT0 = pde.hdiv.assembleB(MESH, space = 'RT0', matrix = 'M', order = 2, shape = MESH.NoEdges)
    
    qK2 = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'K', order = 2)
    qK0 = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'K', order = 0)
    qD1 = pde.l2.assemble(MESH, space = 'P1', matrix = 'M', order = 2)
    # qD0 = pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = 1)
    
    D2 = pde.int.assemble(MESH, order = 2)
    D1 = pde.int.assemble(MESH, order = 1)
    D0 = pde.int.assemble(MESH, order = 0)
    
    
    # circ = pde.int.evaluateB(MESH, order = 0, edges = np.r_[5], like = 0, noint = 1)
    
    # F = D2b@circ@D2b.T
    
    D2b = pde.int.assembleB(MESH, order = 2)
    # D1b = pde.int.assembleB(MESH, order = 1)
    # D0b = pde.int.assembleB(MESH, order = 0)
    
    sigma_outside_eval1 = pde.int.evaluate(MESH, order = 1, coeff = sigma_outside, regions = np.r_[1])
    sigma_circle_eval1  = pde.int.evaluate(MESH, order = 1, coeff = sigma_circle, regions = np.r_[2])
    
    sigma_outside_eval2 = pde.int.evaluate(MESH, order = 2, coeff = sigma_outside, regions = np.r_[1])
    sigma_circle_eval2  = pde.int.evaluate(MESH, order = 2, coeff = sigma_circle, regions = np.r_[2])
    
    Mh2_sigma = qMx@D2@(sigma_outside_eval2 + sigma_circle_eval2)@qMx.T +\
                qMy@D2@(sigma_outside_eval2 + sigma_circle_eval2)@qMy.T
    
    Mh_sigma = qMhx@D1@(sigma_outside_eval1 + sigma_circle_eval1)@qMhx.T +\
               qMhy@D1@(sigma_outside_eval1 + sigma_circle_eval1)@qMhy.T
               
    Mh2_sigma = Mh_sigma.copy()
    
    # epsilon = 1
    Mh_epsilon = qMhx@D1@qMhx.T +\
                 qMhy@D1@qMhy.T
    
    Mh = Mh_epsilon + dt/2*Mh_sigma
    
    K = qK2@D2@qK2.T # C = qD@D1@qK.T
    D1 = qD1@D2@qD1.T
    
    iMh = pde.tools.fastBlockInverse(Mh) # print(sps.linalg.norm(Mh@iMh,np.inf))
    iMh_epsilon = pde.tools.fastBlockInverse(Mh_epsilon) # print(sps.linalg.norm(Mh@iMh,np.inf))
    
    # iMh_Mh_sigma = iMh@Mh_sigma
    # iMh_K = iMh@K
    qMb2_D2b = qMb2@D2b
    qMb2_RT0_D2b = qMb2_RT0@D2b
    
    
    P0,Q0,R0 = reduction_matrix.makeProjectionMatrices(MESH)
    
    circ_DOFS = MESH.Boundary_Edges[MESH.Boundary_Region==5]
    P,Q,R = reduction_matrix.makeProjectionMatrices(MESH, indices = np.sort(circ_DOFS))
          
    # print('K has {:4.2f} MB, iMh has {:4.2f} MB, Mh_sigma has {:4.2f} MB.'.format(\
    #        K.data.nbytes/(1024**2),iMh.data.nbytes/(1024**2),Mh_sigma.data.nbytes/(1024**2)))
    ################################################################################
    
    
    
    ################################################################################
    new_iMh = R@iMh@R.T
    new_Mh_sigma = P.T@Mh2_sigma@P
    new_K = P.T@K@P
    
    new_iMh0 = R0@iMh@R0.T
    new_Mh0_sigma = P0.T@Mh2_sigma@P0
    new_K0 = P0.T@K@P0
    
    print(iMh.shape)
    print(new_iMh.shape)
    print(new_iMh0.shape)
    
    a,_ = sps.linalg.eigs(K,1,Mh_epsilon)
    lam = np.real(a[0])
    tau = np.sqrt(2)/(h_richtig*np.sqrt(lam))
    print('CFL:', tau)
    
    
    # a,_ = sps.linalg.eigs(K,1,Mh_epsilon,maxiter = 10000)
    # lam = np.real(a[0])
    # tau = np.sqrt(2)/(h_richtig*np.sqrt(lam))
    # print('CFL:', tau)
    
    # # C = 0.2; np.real(sps.linalg.eigs(1/2*Mh_epsilon-(C*h_richtig)**2/4*K,1)[0][0])
    
    # for i in range(100):
    #     a,_ = sps.linalg.eigs(K,1,Mh_epsilon+tau*1e10*(R.T@new_Mh_sigma@R-Mh_sigma))
    #     lam = np.real(a[0])
    #     tau = np.sqrt(2)/(h_richtig*np.sqrt(lam))
    # print('new CFL2:', tau)
    
    # tau = 0.1
    
    # v = np.random.rand(Mh_epsilon.shape[0])
    # for i in range(100):
    #     KK = iMh_epsilon@K
    #     v1 = KK@v
    #     v = v1/np.linalg.norm(v1)
    #     lam = (v@KK@v)/(v@v)
    #     tau = np.sqrt(2)/(h_richtig*np.sqrt(lam))
    # print('new CFL3:', tau)
    
    
    
    
    from scipy.optimize import root_scalar
    
    f_min1 = lambda tau : np.real(sps.linalg.eigs(1/2*Mh_epsilon-tau**2/4*K-0*tau/2*(R.T@new_Mh_sigma@R-Mh_sigma),1,which='SR')[0][0])
    root1 = root_scalar(f_min1, bracket=[0.0, 0.5]).root
    print(root1/h_richtig)
    
    f_min1 = lambda tau : np.real(sps.linalg.eigs(1/2*Mh_epsilon-tau**2/4*K-1*tau/2*(R.T@new_Mh_sigma@R-Mh_sigma),1,which='SR')[0][0])
    root1 = root_scalar(f_min1, bracket=[0.0, 0.5]).root
    
    print('kek0 ',root1/h_richtig)
    print('kek1 ',root1/(h_richtig+100/2*h_richtig**2))
    
    print(MESH)
    
    
    
    
    
    # tau = 0.01; 
    # def get_tau(tau):
    #     print(np.real(sps.linalg.eigs(1/2*Mh_epsilon-tau**2/4*K-1*tau/2*(R.T@new_Mh_sigma@R-Mh_sigma),1,which='SR')[0][0]))
    #     print(np.real(sps.linalg.eigs(1/2*Mh_epsilon-tau**2/4*K-0*tau/2*(R.T@new_Mh_sigma@R-Mh_sigma),1,which='SR')[0][0]))
    #     print(tau/h_richtig)
        
    # def get_lam(lam):
    #     print((sps.linalg.eigs(Mh_epsilon-lam*K,1,which='SM')[0][0]))
    #     print(lam)
        
    # v = np.random.rand(Mh_epsilon.shape[0])
    # # tau = 0.4
    # for i in range(100):
    #     # KK = tau*iMh_epsilon@K
    #     KK = tau/(1+tau)*iMh_epsilon@K
    #     v1 = KK@v
    #     v = v1/np.linalg.norm(v1)
    #     lam = (v@KK@v)/(v@v)
    #     # tau = 2/lam
    #     tau = (-lam+np.sqrt(lam**2+8*lam))/(2*lam*h_richtig)
    # print('new CFL4:',tau)
    print('\n\n')
    
    # stop
    
    if i+1!=iterations:
        MESH.refinemesh(); dt = dt/2;
        
        # init_ref = init_ref/(np.sqrt(2)/2); dt = dt/2
        # gmsh.option.setNumber("Mesh.MeshSizeMax", init_ref)
        # gmsh.option.setNumber("Mesh.MeshSizeMin", init_ref)
        # p,e,t,q = pde.petq_generate()
        # MESH = pde.mesh(p,e,t,q)
        
        ################################################################################
        # Shift points to the circle
        ################################################################################
        Indices_PointsOnCircle = np.unique(MESH.EdgesToVertices[MESH.Boundary_Edges[MESH.Boundary_Region==5],:].flatten())
        PointsOnCircle = MESH.p[Indices_PointsOnCircle,:]
        MESH.p[Indices_PointsOnCircle,:] = 0.3*1/np.sqrt(PointsOnCircle[:,0]**2+PointsOnCircle[:,1]**2)[:,None]*PointsOnCircle
        ################################################################################
    
    
gmsh.finalize()
# do()