import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

# import line_profiler
# profile = line_profiler.LineProfiler()

import numpy as np
import pde
# import scipy.sparse as sps
# from sksparse.cholmod import cholesky
import time
import gmsh
import reduction_matrix


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
dt = 0.03/2*4
iterations = 6
init_ref = 0.25*4
use_GPU = False


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
MESH.refinemesh()
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
    print('h approx',h_approx,'dt chosen exactly as ',dt, 'konstante etwa:',dt/h_approx)

    ################################################################################
    qMhx,qMhy = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'M', order = 1)
    qMx,qMy = pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'M', order = 2)
    qMb2 = pde.hdiv.assembleB(MESH, space = 'BDM1', matrix = 'M', order = 2, shape = 2*MESH.NoEdges)
    
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
    
    
    
    ################################################################################
    if not use_GPU:
        tm = time.monotonic()
        
        uh_NC1_oldold = pde.hdiv.interp(MESH, space = 'BDM1', order = 5, f = lambda x,y : np.c_[u1ex(x,y,0),u2ex(x,y,0)])
        uh_NC1_old = pde.hdiv.interp(MESH, space = 'BDM1', order = 5, f = lambda x,y : np.c_[u1ex(x,y,dt),u2ex(x,y,dt)])
        
        
        # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), qMhx.T@uh_NC1_oldold)
        # fig.show()
        
        
        uh_N0_oldold = R@uh_NC1_oldold
        uh_N0_old = R@uh_NC1_old
        
        uh_N00_oldold = R0@uh_NC1_oldold
        uh_N00_old = R0@uh_NC1_old
        
        for j in range(int(T/dt)):
            
            jdt = (j+1)*dt
            
            intF = qMb2_D2b@ pde.int.evaluateB(MESH, order = 2, coeff = lambda x,y : divuex(x,y,jdt), edges = np.r_[1,2,3,4], like = 1)
            
            
            s = iMh.dot(K.dot(uh_NC1_old) + Mh_sigma.dot((uh_NC1_old-uh_NC1_oldold)/dt) + intF)
            uh_NC1 = 2*uh_NC1_old-uh_NC1_oldold-(dt**2)*s
            
            new_intF = R@iMh@intF
            new_s = new_iMh.dot(new_K.dot(uh_N0_old) + new_Mh_sigma.dot((uh_N0_old-uh_N0_oldold)/dt)) + new_intF
            uh_N0 = 2*uh_N0_old-uh_N0_oldold-(dt**2)*new_s
            
            new_intF0 = R0@iMh@intF
            new_s0 = new_iMh0.dot(new_K0.dot(uh_N00_old) + new_Mh0_sigma.dot((uh_N00_old-uh_N00_oldold)/dt)) + new_intF0
            uh_N00 = 2*uh_N00_old-uh_N00_oldold-(dt**2)*new_s0
            
            
            uh_NC1_oldold = uh_NC1_old
            uh_NC1_old = uh_NC1
            
            uh_N0_oldold = uh_N0_old
            uh_N0_old = uh_N0
            
            uh_N00_oldold = uh_N00_old
            uh_N00_old = uh_N00
            
            if (j*100//int(T/dt))%10 == 0:
                print("\rTimestepping : ",j*100//int(T/dt),'%', end = " ")
                # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), qMhx.T@uh_NC1_oldold)
                # fig.show()
        
        print('Time stepping took a total of {:4.8f} seconds.'.format(time.monotonic()-tm))
        print('\n')
    ################################################################################
    
    
    ################################################################################    
    if use_GPU:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
        
        tm = time.monotonic()
        
        uh_NC1_oldold = pde.hdiv.interp(MESH, space = 'BDM1', order = 5, f = lambda x,y : np.c_[u1ex(x,y,0),u2ex(x,y,0)])
        uh_NC1_old = pde.hdiv.interp(MESH, space = 'BDM1', order = 5, f = lambda x,y : np.c_[u1ex(x,y,dt),u2ex(x,y,dt)])
        
        uh_N0_oldold = R@uh_NC1_oldold
        uh_N0_old = R@uh_NC1_old
        
        uh_N00_oldold = R0@uh_NC1_oldold
        uh_N00_old = R0@uh_NC1_old
        
        tp = cp.float64
        
        cuda_uh_NC1_oldold = cp.array(uh_NC1_oldold, dtype = tp)
        cuda_uh_NC1_old = cp.array(uh_NC1_old, dtype = tp)
        
        cuda_uh_N0_oldold = cp.array(uh_N0_oldold, dtype = tp)
        cuda_uh_N0_old = cp.array(uh_N0_old, dtype = tp)
        
        cuda_uh_N00_oldold = cp.array(uh_N00_oldold, dtype = tp)
        cuda_uh_N00_old = cp.array(uh_N00_old, dtype = tp)
        
        cuda_K = cp_csr_matrix(K, dtype = tp)
        cuda_iMh = cp_csr_matrix(iMh, dtype = tp)
        cuda_Mh_sigma = cp_csr_matrix(Mh2_sigma, dtype = tp)
        
        cuda_new_iMh = cp_csr_matrix(new_iMh, dtype = tp)
        cuda_new_Mh_sigma = cp_csr_matrix(new_Mh_sigma, dtype = tp)
        cuda_new_K = cp_csr_matrix(new_K, dtype = tp)
        
        cuda_new_iMh0 = cp_csr_matrix(new_iMh0, dtype = tp)
        cuda_new_Mh0_sigma = cp_csr_matrix(new_Mh0_sigma, dtype = tp)
        cuda_new_K0 = cp_csr_matrix(new_K0, dtype = tp)
        
        cuda_R = cp_csr_matrix(R, dtype = tp)
        cuda_R0 = cp_csr_matrix(R0, dtype = tp)
        
        for j in range(int(T/dt)):
            jdt = j*dt
            
            intF = qMb2_D2b@ pde.int.evaluateB(MESH, order = 2, coeff = lambda x,y : divuex(x,y,jdt), edges = np.r_[1,2,3,4], like = 1)
                    
            cuda_intF = cp.array(intF, dtype = tp)
            
            s = cuda_iMh.dot(cuda_K.dot(cuda_uh_NC1_old) + cuda_Mh_sigma.dot((cuda_uh_NC1_old-cuda_uh_NC1_oldold)/dt) + cuda_intF)
            cuda_uh_NC1 = 2*cuda_uh_NC1_old-cuda_uh_NC1_oldold-(dt**2)*s
                    
            cuda_new_intF = cuda_R@cuda_iMh@cuda_intF
            cuda_new_s = cuda_new_iMh.dot(cuda_new_K.dot(cuda_uh_N0_old) + cuda_new_Mh_sigma.dot((cuda_uh_N0_old-cuda_uh_N0_oldold)/dt)) + cuda_new_intF
            cuda_uh_N0 = 2*cuda_uh_N0_old-cuda_uh_N0_oldold-(dt**2)*cuda_new_s
            
            cuda_new_intF0 = cuda_R0@cuda_iMh@cuda_intF
            cuda_new_s0 = cuda_new_iMh0.dot(cuda_new_K0.dot(cuda_uh_N00_old) + cuda_new_Mh0_sigma.dot((cuda_uh_N00_old-cuda_uh_N00_oldold)/dt)) + cuda_new_intF0
            cuda_uh_N00 = 2*cuda_uh_N00_old-cuda_uh_N00_oldold-(dt**2)*cuda_new_s0
            
            cuda_uh_NC1_oldold = cuda_uh_NC1_old
            cuda_uh_NC1_old = cuda_uh_NC1
            
            cuda_uh_N0_oldold = cuda_uh_N0_old
            cuda_uh_N0_old = cuda_uh_N0
            
            cuda_uh_N00_oldold = cuda_uh_N00_old
            cuda_uh_N00_old = cuda_uh_N00
            
            if (j*100//int(T/dt))%10 == 0:
                print("\rTimestepping : ",j*100//int(T/dt),'%', end = " ")
            
            
        uh_NC1 = cp.ndarray.get(cuda_uh_NC1)
        uh_N0 = cp.ndarray.get(cuda_uh_N0)
        uh_N00 = cp.ndarray.get(cuda_uh_N00)
        
        # print(np.linalg.norm(uh_N0))
        
        uh_NC1_oldold = cp.ndarray.get(cuda_uh_NC1_oldold)
        uh_N0_oldold = cp.ndarray.get(cuda_uh_N0_oldold)
        uh_N00_oldold = cp.ndarray.get(cuda_uh_N00_oldold)
        
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
    
    if i>0:
        error[i] = np.sqrt((dtau_uh_x_P1d-dtau_uh_x_P1d_fine)@D1@(dtau_uh_x_P1d-dtau_uh_x_P1d_fine)+\
                           (dtau_uh_y_P1d-dtau_uh_y_P1d_fine)@D1@(dtau_uh_y_P1d-dtau_uh_y_P1d_fine)+\
                           (mean_div_uh_NC1_P0-mean_div_uh_NC1_P0_fine)@D0@(mean_div_uh_NC1_P0-mean_div_uh_NC1_P0_fine))
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
    
    
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), uh_x_P1d)
    # fig.show()
    
    if i>0:
        
        new_error[i] = np.sqrt((dtau_new_uh_x_P1d-dtau_new_uh_x_P1d_fine)@D1@(dtau_new_uh_x_P1d-dtau_new_uh_x_P1d_fine)+\
                               (dtau_new_uh_y_P1d-dtau_new_uh_y_P1d_fine)@D1@(dtau_new_uh_y_P1d-dtau_new_uh_y_P1d_fine)+\
                               (mean_new_div_uh_N0_P0-mean_new_div_uh_N0_P0_fine)@D0@(mean_new_div_uh_N0_P0-mean_new_div_uh_N0_P0_fine))
    
    
        error2[i] = np.sqrt((dtau_new_uh_x_P1d-dtau_uh_x_P1d)@D1@(dtau_new_uh_x_P1d-dtau_uh_x_P1d)+\
                            (dtau_new_uh_y_P1d-dtau_uh_y_P1d)@D1@(dtau_new_uh_y_P1d-dtau_uh_y_P1d)+\
                            (mean_new_div_uh_N0_P0-mean_div_uh_NC1_P0)@D0@(mean_new_div_uh_N0_P0-mean_div_uh_NC1_P0))
        
        
        error3[i] = np.sqrt((dtau_new_uh0_x_P1d-dtau_uh_x_P1d)@D1@(dtau_new_uh0_x_P1d-dtau_uh_x_P1d)+\
                            (dtau_new_uh0_y_P1d-dtau_uh_y_P1d)@D1@(dtau_new_uh0_y_P1d-dtau_uh_y_P1d)+\
                            (mean_new_div_uh_N00_P0-mean_div_uh_NC1_P0)@D0@(mean_new_div_uh_N00_P0-mean_div_uh_NC1_P0))
        
        # error3[i] = np.sqrt((new_uh0_x_P1d-new_uh_x_P1d)@D1@(new_uh0_x_P1d-new_uh_x_P1d)+\
        #                     (new_uh0_y_P1d-new_uh_y_P1d)@D1@(new_uh0_y_P1d-new_uh_y_P1d))
        
        # v = (dtau_new_uh0_x_P1d-dtau_new_uh_x_P1d)**2 + (dtau_new_uh0_y_P1d-dtau_new_uh_y_P1d)**2
        # print('max value is:',np.max(v))
        
        
        # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), (dtau_new_uh0_x_P1d-dtau_new_uh_x_P1d)**2 + (dtau_new_uh0_y_P1d-dtau_new_uh_y_P1d)**2)
        # fig.show()
    ################################################################################

    dtau_uh_x_P1d_fine = MESH.refine(dtau_uh_x_P1d)
    dtau_uh_y_P1d_fine = MESH.refine(dtau_uh_y_P1d)
    mean_div_uh_NC1_P0_fine = MESH.refine(mean_div_uh_NC1_P0)
    
    dtau_new_uh_x_P1d_fine = MESH.refine(dtau_new_uh_x_P1d)
    dtau_new_uh_y_P1d_fine = MESH.refine(dtau_new_uh_y_P1d)
    mean_new_div_uh_N0_P0_fine = MESH.refine(mean_new_div_uh_N0_P0)
    
    dtau_new_uh0_x_P1d_fine = MESH.refine(dtau_new_uh0_x_P1d)
    dtau_new_uh0_y_P1d_fine = MESH.refine(dtau_new_uh0_y_P1d)
    mean_new_div_uh_N00_P0_fine = MESH.refine(mean_new_div_uh_N00_P0)
    
    ################################################################################
    
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), np.sqrt(uh_x_P1d**2+uh_y_P1d**2), u_height=0)
    # fig.data[0].colorscale='Jet'
    # fig.show()
    
    # fig = MESH.pdesurf_hybrid(dict(trig = 'P1d', controls = 1), np.sqrt(new_uh_x_P1d**2+new_uh_y_P1d**2), u_height=0)
    # fig.data[0].colorscale='Jet'
    # fig.show()
    
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
    
    
    rate = np.log2(error[1:-1]/error[2:])
    print("Convergenge rates : ",rate)
    # print("Errors: ",error)
    
    new_rate = np.log2(new_error[1:-1]/new_error[2:])
    print("Convergenge rates : ",new_rate)
    # print("Errors: ",new_error)
    
    rate2 = np.log2(error2[1:-1]/error2[2:])
    print("Convergenge rates : ",rate2)
    # print("Errors: ",error2)
    
    rate3 = np.log2(error3[1:-1]/error3[2:])
    print("Convergenge rates : ",rate3)
    # print("Errors: ",error3)
    
gmsh.finalize()
# do()