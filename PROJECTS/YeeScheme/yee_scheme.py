import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

# import line_profiler
# profile = line_profiler.LineProfiler()

from sparse_dot_mkl import dot_product_mkl as mult
import numpy as np
import gc
import pde
# import scipy.sparse as sps
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

T = 2.5
dt = 0.03/2*2*2/2*1.5
iterations = 5#6
init_ref = 0.125#0.25
use_GPU = True

def to_torch_csr(X):
    # coo = X.tocoo()
    # row = torch.from_numpy(coo.row.astype(np.int64)).to(torch.long)
    # col = torch.from_numpy(coo.col.astype(np.int64)).to(torch.long)
    # edge_index = torch.stack([row, col], dim=0)
    # val = torch.from_numpy(coo.data.astype(np.float64))
    # return torch.sparse.FloatTensor(edge_index, val, torch.Size(coo.shape)).to_sparse_csr()#.cuda()
    return X

def torch_from_numpy(X):
    # return torch.from_numpy(X)#.cuda()
    return X

def numpy_from_torch(X):
    # return X.cpu().numpy()
    # return X.numpy()
    return X



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
new_error2 = np.zeros(iterations)
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
    
    # iMh_Mh_sigma = iMh@Mh_sigma
    # iMh_K = iMh@K
    qMb2_D2b = qMb2@D2b
    qMb2_RT0_D2b = qMb2_RT0@D2b
    
    
    P0,Q0,R0 = reduction_matrix.makeProjectionMatrices(MESH)
    
    circ_DOFS = MESH.Boundary_Edges[MESH.Boundary_Region==5]
    P,Q,R = reduction_matrix.makeProjectionMatrices(MESH, indices = np.sort(circ_DOFS))
    
    # P = P0.copy()
    # Q = Q0.copy()
    # R = R0.copy()
          
    print('K has {:4.2f} MB, iMh has {:4.2f} MB, Mh_sigma has {:4.2f} MB.'.format(\
           K.data.nbytes/(1024**2),iMh.data.nbytes/(1024**2),Mh_sigma.data.nbytes/(1024**2)))
    ################################################################################
    
    
    
    ################################################################################
    new_iMh = R@iMh@R.T
    new_Mh_sigma = P.T@Mh2_sigma@P
    new_K = P.T@K@P
    
    new_iMh0 = R0@iMh@R0.T
    new_Mh0_sigma = P0.T@Mh2_sigma@P0
    new_K0 = P0.T@K@P0
    ################################################################################
    
    uh_NC1_oldold = pde.hdiv.interp(MESH, space = 'BDM1', order = 5, f = lambda x,y : np.c_[u1ex(x,y,0),u2ex(x,y,0)])
    uh_NC1_old = pde.hdiv.interp(MESH, space = 'BDM1', order = 5, f = lambda x,y : np.c_[u1ex(x,y,dt),u2ex(x,y,dt)])
    
    uh_N0_oldold = R@uh_NC1_oldold
    uh_N0_old = R@uh_NC1_old
    
    uh_N00_oldold = R0@uh_NC1_oldold
    uh_N00_old = R0@uh_NC1_old
    
    ################################################################################
    if not use_GPU:
        tm = time.monotonic()
        
        for j in range(int(T/dt)):
            
            jdt = (j+1)*dt
            
            intF = qMb2_D2b@ pde.int.evaluateB(MESH, order = 2, coeff = lambda x,y : divuex(x,y,jdt), edges = np.r_[1,2,3,4], like = 1)
            intF_RT0 = qMb2_RT0_D2b@ pde.int.evaluateB(MESH, order = 2, coeff = lambda x,y : divuex(x,y,jdt), edges = np.r_[1,2,3,4], like = 1)
            
            s = iMh.dot(K.dot(uh_NC1_old) + Mh_sigma.dot((uh_NC1_old-uh_NC1_oldold)/dt) + intF)
            uh_NC1 = 2*uh_NC1_old-uh_NC1_oldold-(dt**2)*s
            
            # new_intF = R@iMh@R0.T@intF_RT0
            new_intF = R@iMh@intF
            new_s = new_iMh.dot(new_K.dot(uh_N0_old) + new_Mh_sigma.dot((uh_N0_old-uh_N0_oldold)/dt)) + new_intF
            uh_N0 = 2*uh_N0_old-uh_N0_oldold-(dt**2)*new_s
            
            # new_intF0 = R0@iMh@R0.T@intF_RT0
            new_intF0 = R0@iMh@intF
            new_s0 = new_iMh0.dot(new_K0.dot(uh_N00_old) + new_Mh0_sigma.dot((uh_N00_old-uh_N00_oldold)/dt)) + new_intF0
            uh_N00 = 2*uh_N00_old-uh_N00_oldold-(dt**2)*new_s0
            
            uh_NC1_oldold = uh_NC1_old
            uh_NC1_old = uh_NC1
            
            uh_N0_oldold = uh_N0_old
            uh_N0_old = uh_N0
            
            uh_N00_oldold = uh_N00_old
            uh_N00_old = uh_N00
            
            # if (j*100//int(T/dt))%10 == 0:
            if (i==5):
                if (np.abs(j*dt-1.99)<dt) or (np.abs(j*dt-1)<dt) or (np.abs(j*dt-1.5)<dt):
                    print("\rTimestepping : ",j*100//int(T/dt),'%', end = " ")
                    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), qMhx.T@uh_NC1_oldold)
                    # fig.show()
                    # uh_x_P1d = qMhx.T@uh_NC1
                    # uh_y_P1d = qMhy.T@uh_NC1
                    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), np.sqrt(uh_x_P1d**2+uh_y_P1d**2), u_height=0)
                    # fig.data[0].colorscale='Jet'
                    # fig.data[0].cmax = 2.5
                    # fig.data[0].cmin = 0
                    # fig.show()
        
        print('Time stepping took a total of {:4.8f} seconds.'.format(time.monotonic()-tm))
        print('\n')
    ################################################################################
    
    
    ################################################################################    
    if use_GPU:
        # import cupy as cp
        # from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
        # import torch
        # torch.set_num_threads(1)
        # torch.no_grad()
        
        # torch.cuda.set_device(1)
        
        tm = time.monotonic()
        
        # to_torch_csr(X)
        
        cuda_uh_NC1_oldold = torch_from_numpy(uh_NC1_oldold)
        cuda_uh_NC1_old = torch_from_numpy(uh_NC1_old)
        cuda_uh_NC1 = torch_from_numpy(uh_NC1_old)
        
        cuda_uh_N0_oldold = torch_from_numpy(uh_N0_oldold)
        cuda_uh_N0_old = torch_from_numpy(uh_N0_old)
        cuda_uh_N0 = torch_from_numpy(uh_N0_old)
        
        cuda_uh_N00_oldold = torch_from_numpy(uh_N00_oldold)
        cuda_uh_N00_old = torch_from_numpy(uh_N00_old)
        cuda_uh_N00 = torch_from_numpy(uh_N00_old)
        
        cuda_K = to_torch_csr(K)
        cuda_iMh = to_torch_csr(iMh)
        cuda_Mh_sigma = to_torch_csr(Mh_sigma)
        
        cuda_new_iMh = to_torch_csr(new_iMh)
        cuda_new_Mh_sigma = to_torch_csr(new_Mh_sigma)
        cuda_new_K = to_torch_csr(new_K)
        
        cuda_new_iMh0 = to_torch_csr(new_iMh0)
        cuda_new_Mh0_sigma = to_torch_csr(new_Mh0_sigma)
        cuda_new_K0 = to_torch_csr(new_K0)
        
        cuda_R = to_torch_csr(R)
        cuda_R0 = to_torch_csr(R0)
        
        
        for j in range(int(T/dt)):
            jdt = (j+1)*dt
            
            del cuda_uh_NC1
            del cuda_uh_N0
            del cuda_uh_N00
            
            intF = qMb2_D2b@ pde.int.evaluateB(MESH, order = 2, coeff = lambda x,y : divuex(x,y,jdt), edges = np.r_[1,2,3,4], like = 1)            
            intF_RT0 = qMb2_RT0_D2b@ pde.int.evaluateB(MESH, order = 2, coeff = lambda x,y : divuex(x,y,jdt), edges = np.r_[1,2,3,4], like = 1)
            
            cuda_intF = torch_from_numpy(intF)
            cuda_intF_RT0 = torch_from_numpy(intF_RT0)
            
            s = mult(cuda_iMh,mult(cuda_K,cuda_uh_NC1_old) + mult(cuda_Mh_sigma,(cuda_uh_NC1_old-cuda_uh_NC1_oldold)/dt) + cuda_intF)
            cuda_uh_NC1 = (2*cuda_uh_NC1_old-cuda_uh_NC1_oldold-(dt**2)*s)
                    
            # cuda_new_intF = mult(cuda_R,mult(cuda_iMh,mult(cuda_R.T,cuda_intF_RT0)))
            cuda_new_intF = mult(cuda_R,mult(cuda_iMh,cuda_intF))
            cuda_new_s = mult(cuda_new_iMh,mult(cuda_new_K,cuda_uh_N0_old) + mult(cuda_new_Mh_sigma,(cuda_uh_N0_old-cuda_uh_N0_oldold)/dt)) + cuda_new_intF
            cuda_uh_N0 = (2*cuda_uh_N0_old-cuda_uh_N0_oldold-(dt**2)*cuda_new_s)
            
            
            # cuda_new_intF0 = mult(cuda_R0,mult(cuda_iMh,mult(cuda_R0.T,cuda_intF_RT0)))
            cuda_new_intF0 = mult(cuda_R0,mult(cuda_iMh,cuda_intF))
            cuda_new_s0 = mult(cuda_new_iMh0,mult(cuda_new_K0,cuda_uh_N00_old) + mult(cuda_new_Mh0_sigma,(cuda_uh_N00_old-cuda_uh_N00_oldold)/dt)) + cuda_new_intF0
            cuda_uh_N00 = (2*cuda_uh_N00_old-cuda_uh_N00_oldold-(dt**2)*cuda_new_s0)
            
            # del cuda_intF
            # del cuda_new_intF
            # del cuda_new_intF0
            
            # del cuda_uh_NC1_oldold
            cuda_uh_NC1_oldold = cuda_uh_NC1_old
            # del cuda_uh_NC1_old
            cuda_uh_NC1_old = cuda_uh_NC1
            
            # del cuda_uh_N0_oldold
            cuda_uh_N0_oldold = cuda_uh_N0_old
            # del cuda_uh_N0_old
            cuda_uh_N0_old = cuda_uh_N0
            
            # del cuda_uh_N00_oldold
            cuda_uh_N00_oldold = cuda_uh_N00_old
            # del cuda_uh_N00_old
            cuda_uh_N00_old = cuda_uh_N00
            
            # del cuda_new_s
            # del cuda_new_s0
            
            
            if (j*100//int(T/dt))%10 == 0:
                print("\rTimestepping : ",j*100//int(T/dt),'%', end = " ")
                
            
        # del cuda_K
        # del cuda_iMh
        # del cuda_Mh_sigma
        
        # del cuda_new_iMh
        # del cuda_new_Mh_sigma
        # del cuda_new_K
        
        # del cuda_new_iMh0
        # del cuda_new_Mh0_sigma
        # del cuda_new_K0
        
        # del cuda_R
        # del cuda_R0
        
        uh_NC1 = numpy_from_torch(cuda_uh_NC1)
        uh_N0 = numpy_from_torch(cuda_uh_N0)
        uh_N00 = numpy_from_torch(cuda_uh_N00)
        
        print('ACHTUNG: ', cuda_uh_NC1.dtype,'\n')
        
        uh_NC1_oldold = numpy_from_torch(cuda_uh_NC1_oldold)
        uh_N0_oldold = numpy_from_torch(cuda_uh_N0_oldold)
        uh_N00_oldold = numpy_from_torch(cuda_uh_N00_oldold)
        
        print('Time stepping took a total of {:4.8f} seconds.'.format(time.monotonic()-tm))
        print('\n')
    ################################################################################
    
    
    
    ################################################################################
    uh_x_P1d = qMhx.T@uh_NC1
    uh_y_P1d = qMhy.T@uh_NC1
    
    uh_x_P1d_old = qMhx.T@uh_NC1_oldold
    uh_y_P1d_old = qMhy.T@uh_NC1_oldold    
    
    div_uh_NC1_P0 = qK0.T@uh_NC1
    div_uh_NC1_P0_old = qK0.T@uh_NC1_oldold
    
    mean_div_uh_NC1_P0 = 1/2*(div_uh_NC1_P0 + div_uh_NC1_P0_old)
    
    dtau_uh_x_P1d = 1/dt*(uh_x_P1d-uh_x_P1d_old)
    dtau_uh_y_P1d = 1/dt*(uh_y_P1d-uh_y_P1d_old)
    
    ################################################################################
    
    
    ################################################################################
    
    
    prol_uh_N0 = P@uh_N0
    prol_uh_N0_oldold = P@uh_N0_oldold
    
    new_uh_x_P1d = qMhx.T@prol_uh_N0
    new_uh_y_P1d = qMhy.T@prol_uh_N0
    
    new_uh_x_P1d_old = qMhx.T@prol_uh_N0_oldold
    new_uh_y_P1d_old = qMhy.T@prol_uh_N0_oldold
    
    new_div_uh_N0_P0 = qK0.T@prol_uh_N0
    new_div_uh_N0_P0_old = qK0.T@prol_uh_N0_oldold
    
    mean_new_div_uh_N0_P0 = 1/2*(new_div_uh_N0_P0 + new_div_uh_N0_P0_old)
    
    dtau_new_uh_x_P1d = 1/dt*(new_uh_x_P1d-new_uh_x_P1d_old)
    dtau_new_uh_y_P1d = 1/dt*(new_uh_y_P1d-new_uh_y_P1d_old)
    
    
    
    prol_uh_N00 = P0@uh_N00
    prol_uh_N00_oldold = P0@uh_N00_oldold
    
    new_uh0_x_P1d = qMhx.T@prol_uh_N00
    new_uh0_y_P1d = qMhy.T@prol_uh_N00
    
    new_uh0_x_P1d_old = qMhx.T@prol_uh_N00_oldold
    new_uh0_y_P1d_old = qMhy.T@prol_uh_N00_oldold
    
    new_div_uh_N00_P0 = qK0.T@prol_uh_N00
    new_div_uh_N00_P0_old = qK0.T@prol_uh_N00_oldold
    
    mean_new_div_uh_N00_P0 = 1/2*(new_div_uh_N00_P0 + new_div_uh_N00_P0_old)
    
    dtau_new_uh0_x_P1d = 1/dt*(new_uh0_x_P1d-new_uh0_x_P1d_old)
    dtau_new_uh0_y_P1d = 1/dt*(new_uh0_y_P1d-new_uh0_y_P1d_old)
    
    # torch.cuda.empty_cache()
    
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), uh_x_P1d)
    # fig.show()
    
    if i>0:
    
        loc,_ = MESH._mesh__ismember(MESH.TriangleToEdges,MESH.Boundary_Edges[MESH.Boundary_Region==5])
        trig_edge = np.argwhere(loc)[:,0]
        C1 = pde.int.evaluate(MESH, order = 1, coeff = lambda x,y : 1+0*x*y, indices = trig_edge)
        C0 = pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : 1+0*x*y, indices = trig_edge)
        
        error[i] = np.sqrt((dtau_uh_x_P1d-dtau_uh_x_P1d_fine)@D1@(dtau_uh_x_P1d-dtau_uh_x_P1d_fine))/np.sqrt((dtau_uh_x_P1d)@D1@(dtau_uh_x_P1d))+\
                   np.sqrt((dtau_uh_y_P1d-dtau_uh_y_P1d_fine)@D1@(dtau_uh_y_P1d-dtau_uh_y_P1d_fine))/np.sqrt((dtau_uh_y_P1d)@D1@(dtau_uh_y_P1d))+\
                   np.sqrt((mean_div_uh_NC1_P0-mean_div_uh_NC1_P0_fine)@D0@(mean_div_uh_NC1_P0-mean_div_uh_NC1_P0_fine))/np.sqrt((mean_div_uh_NC1_P0)@D0@(mean_div_uh_NC1_P0))
        
        # reduziert nur auf circle net
        new_error[i] = np.sqrt((dtau_new_uh_x_P1d-dtau_new_uh_x_P1d_fine)@D1@(dtau_new_uh_x_P1d-dtau_new_uh_x_P1d_fine))+\
                       np.sqrt((dtau_new_uh_y_P1d-dtau_new_uh_y_P1d_fine)@D1@(dtau_new_uh_y_P1d-dtau_new_uh_y_P1d_fine))+\
                       np.sqrt((mean_new_div_uh_N0_P0-mean_new_div_uh_N0_P0_fine)@D0@(mean_new_div_uh_N0_P0-mean_new_div_uh_N0_P0_fine))
                       
        # reduziert überall
        new_error2[i] = np.sqrt((dtau_new_uh0_x_P1d-dtau_new_uh0_x_P1d_fine)@D1@C1@(dtau_new_uh0_x_P1d-dtau_new_uh0_x_P1d_fine))+\
                        np.sqrt((dtau_new_uh0_y_P1d-dtau_new_uh0_y_P1d_fine)@D1@C1@(dtau_new_uh0_y_P1d-dtau_new_uh0_y_P1d_fine))+\
                        np.sqrt((mean_new_div_uh_N00_P0-mean_new_div_uh_N00_P0_fine)@D0@C0@(mean_new_div_uh_N00_P0-mean_new_div_uh_N00_P0_fine))
    
        # 0 - reduce everywhere except the circle
        error2[i] = np.sqrt((dtau_new_uh_x_P1d-dtau_uh_x_P1d)@D1@(dtau_new_uh_x_P1d-dtau_uh_x_P1d))/np.sqrt((dtau_uh_x_P1d)@D1@(dtau_uh_x_P1d))+\
                    np.sqrt((dtau_new_uh_y_P1d-dtau_uh_y_P1d)@D1@(dtau_new_uh_y_P1d-dtau_uh_y_P1d))/np.sqrt((dtau_uh_y_P1d)@D1@(dtau_uh_y_P1d))+\
                    np.sqrt((mean_new_div_uh_N0_P0-mean_div_uh_NC1_P0)@D0@(mean_new_div_uh_N0_P0-mean_div_uh_NC1_P0))/np.sqrt((mean_div_uh_NC1_P0)@D0@(mean_div_uh_NC1_P0))
        
        # 00 - reduce everywhere
        error3[i] = np.sqrt((dtau_new_uh0_x_P1d-dtau_uh_x_P1d)@D1@(dtau_new_uh0_x_P1d-dtau_uh_x_P1d))/np.sqrt((dtau_uh_x_P1d)@D1@(dtau_uh_x_P1d))+\
                    np.sqrt((dtau_new_uh0_y_P1d-dtau_uh_y_P1d)@D1@(dtau_new_uh0_y_P1d-dtau_uh_y_P1d))/np.sqrt((dtau_uh_y_P1d)@D1@(dtau_uh_y_P1d))+\
                    np.sqrt((mean_new_div_uh_N00_P0-mean_div_uh_NC1_P0)@D0@(mean_new_div_uh_N00_P0-mean_div_uh_NC1_P0))/np.sqrt((mean_div_uh_NC1_P0)@D0@(mean_div_uh_NC1_P0))
                    
        # 00 - reduce everywhere
        # error3[i] = np.sqrt((dtau_new_uh0_x_P1d-dtau_uh_x_P1d)@D1@(dtau_new_uh0_x_P1d-dtau_uh_x_P1d))+\
        #             np.sqrt((dtau_new_uh0_y_P1d-dtau_uh_y_P1d)@D1@(dtau_new_uh0_y_P1d-dtau_uh_y_P1d))+\
        #             np.sqrt((mean_new_div_uh_N00_P0-mean_div_uh_NC1_P0)@D0@(mean_new_div_uh_N00_P0-mean_div_uh_NC1_P0))
        # error3[i] = np.sqrt((new_uh0_x_P1d-new_uh_x_P1d)@D1@(new_uh0_x_P1d-new_uh_x_P1d)+\
        #                     (new_uh0_y_P1d-new_uh_y_P1d)@D1@(new_uh0_y_P1d-new_uh_y_P1d))
        
        # v = (dtau_new_uh0_x_P1d-dtau_new_uh_x_P1d)**2 + (dtau_new_uh0_y_P1d-dtau_new_uh_y_P1d)**2
        # print('max value is:',np.max(v))
        
        
        # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), (C1.diagonal())*((dtau_new_uh0_x_P1d-dtau_new_uh0_x_P1d_fine)**2 + (dtau_new_uh0_y_P1d-dtau_new_uh0_y_P1d_fine)**2), u_height = 1) # da passiert was
        fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), np.sqrt((new_uh_x_P1d)**2 + (new_uh_y_P1d)**2), u_height = 0) # da passiert was
        # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), (dtau_new_uh_x_P1d-dtau_new_uh0_x_P1d)**2 + (dtau_new_uh_y_P1d-dtau_new_uh0_y_P1d)**2, u_height = 1) # da passiert was
        # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), np.abs(    dtau_uh_x_P1d-dtau_new_uh0_x_P1d), u_height = 0)
        # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), np.abs(    dtau_uh_x_P1d-dtau_new_uh_x_P1d), u_height = 0)
        # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), (dtau_new_uh0_x_P1d-dtau_new_uh_x_P1d)**2 + (dtau_new_uh0_y_P1d-dtau_new_uh_y_P1d)**2)
        
        fig.data[0].colorscale='Jet'
        fig.data[0].cmax=2.5
        fig.show()
    ################################################################################
    
    ################################################################################
    
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), np.sqrt(uh_x_P1d**2+uh_y_P1d**2), u_height=0)
    # fig.data[0].colorscale='Jet'
    # fig.show()
    
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), np.sqrt(new_uh_x_P1d**2+new_uh_y_P1d**2), u_height=0)
    # fig.data[0].colorscale='Jet'
    # fig.show()
    
    if i+1!=iterations:

        dtau_uh_x_P1d_fine = MESH.refine(dtau_uh_x_P1d)
        dtau_uh_y_P1d_fine = MESH.refine(dtau_uh_y_P1d)
        mean_div_uh_NC1_P0_fine = MESH.refine(mean_div_uh_NC1_P0)
        
        dtau_new_uh_x_P1d_fine = MESH.refine(dtau_new_uh_x_P1d)
        dtau_new_uh_y_P1d_fine = MESH.refine(dtau_new_uh_y_P1d)
        mean_new_div_uh_N0_P0_fine = MESH.refine(mean_new_div_uh_N0_P0)
        
        dtau_new_uh0_x_P1d_fine = MESH.refine(dtau_new_uh0_x_P1d)
        dtau_new_uh0_y_P1d_fine = MESH.refine(dtau_new_uh0_y_P1d)
        mean_new_div_uh_N00_P0_fine = MESH.refine(mean_new_div_uh_N00_P0)
        
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
    
    
    rate = np.log2(error[1:-1]/error[2:])
    print("1. Convergenge rates : ",rate)
    print("1. Errors: ",error)
    
    new_rate = np.log2(new_error[1:-1]/new_error[2:])
    print("2. Convergenge rates : ",new_rate)
    print("2. Errors: ",new_error)
    
    new_rate2 = np.log2(new_error2[1:-1]/new_error2[2:])
    print("3. Convergenge rates : ",new_rate2)
    print("3. Errors: ",new_error2)
    
    rate2 = np.log2(error2[1:-1]/error2[2:])
    print("4. Convergenge rates : ",rate2)
    print("4. Errors: ",error2)
    
    rate3 = np.log2(error3[1:-1]/error3[2:])
    print("5. Convergenge rates : ",rate3)
    print("5. Errors: ",error3)
    
gmsh.finalize()
# do()