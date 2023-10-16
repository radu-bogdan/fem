import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

# import line_profiler
# profile = line_profiler.LineProfiler()

# from sparse_dot_mkl import dot_product_mkl as mult
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

T = 2
dt = 0.03/2*2*1.8
iterations = 2
init_ref = 0.25
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

    
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), np.sqrt(uh_x_P1d**2+uh_y_P1d**2), u_height=0)
    # fig.data[0].colorscale='Jet'
    # fig.show()
    
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), np.sqrt(new_uh_x_P1d**2+new_uh_y_P1d**2), u_height=0)
    # fig.data[0].colorscale='Jet'
    # fig.show()
    
    print(iMh.shape)
    print(new_iMh.shape)
    print(new_iMh0.shape)
    print('\n\n')
    
    if i+1!=iterations:
        MESH.refinemesh(); dt = dt/2;
        
        
        loc,_ = MESH._mesh__ismember(MESH.TriangleToEdges,MESH.Boundary_Edges[MESH.Boundary_Region==5])
        trig_edge = np.argwhere(loc)[:,0]
        C = pde.int.evaluate(MESH, order = 1, coeff = lambda x,y : 1+0*x*y, indices = trig_edge)
        
        fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), C.diagonal(), u_height = 1) # da passiert was
        fig.show()
        # init_ref = init_ref/(np.sqrt(2)/2); dt = dt/2
        # gmsh.option.setNumber("Mesh.MeshSizeMax", init_ref)
        # gmsh.option.setNumber("Mesh.MeshSizeMin", init_ref)
        # p,e,t,q = pde.petq_generate()
        # MESH = pde.mesh(p,e,t,q)
        
        ################################################################################
        # Shift points to the circle
        ################################################################################
        Indices_PointsOnCircle = np.unique(MESH.EdgesToVertices[MESH.Boundary_Edges[MESH.Boundary_Region==5],:2].flatten())
        PointsOnCircle = MESH.p[Indices_PointsOnCircle,:]
        MESH.p[Indices_PointsOnCircle,:] = 0.3*1/np.sqrt(PointsOnCircle[:,0]**2+PointsOnCircle[:,1]**2)[:,None]*PointsOnCircle
        ################################################################################
    
    
gmsh.finalize()
# do()