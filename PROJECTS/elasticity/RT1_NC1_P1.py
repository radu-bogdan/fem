import sys
sys.path.insert(0,'../../') # adds parent directory

import numpy as np
import gmsh
import pde
import matplotlib.pyplot as plt
import scipy.sparse as sp

import plotly.io as pio
pio.renderers.default = 'browser'
np.set_printoptions(precision = 8)

lam = 123
mu = 79.3

c1 = (1/(2*mu)-1/(2*mu)*lam/(2*mu+2*lam))**(-1)
c2 = (1/(2*mu))**(-1)

u1_test = lambda x,y : np.sin(np.pi*x)*np.sin(np.pi*y)
u2_test = lambda x,y : np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

u1x_test = lambda x,y : np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
u1y_test = lambda x,y : np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
u2x_test = lambda x,y : 2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
u2y_test = lambda x,y : 2*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

sigma11 = lambda x,y : c1*(np.pi*np.cos(np.pi*x)*np.sin(np.pi*y))
sigma12 = lambda x,y : c2*(1/2*(  np.pi*np.sin(np.pi*x)*np.cos(np.pi*y) +\
                                2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)))
sigma21 = lambda x,y : c2*(1/2*(  np.pi*np.sin(np.pi*x)*np.cos(np.pi*y) +\
                                2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)))
sigma22 = lambda x,y : c1*(2*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y))

divsigma1 = lambda x,y : 1/2*np.pi**2*(4*c2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)\
                                     -(2*c1+c2)*np.sin(np.pi*x)*np.sin(np.pi*y))
divsigma2 = lambda x,y : 1/2*np.pi**2*(c2*np.cos(np.pi*x)*np.cos(np.pi*y)\
                                   -4*(2*c1+c2)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))
    
p_test = lambda x,y : 1/2*(u1y_test(x,y)-u2x_test(x,y))



init_ref = 1
iterations = 5

################################################################################
gmsh.initialize()
gmsh.open('squareg.geo')
gmsh.option.setNumber("Mesh.Algorithm", 1)
gmsh.option.setNumber("Mesh.MeshSizeMax", init_ref)
gmsh.option.setNumber("Mesh.MeshSizeMin", init_ref)
gmsh.option.setNumber("Mesh.SaveAll", 1)
# gmsh.fltk.run()
p,e,t,q = pde.petq_generate()
# gmsh.write("unit_square.m")
gmsh.finalize()

MESH = pde.mesh(p,e,t,q)
################################################################################


################################################################################

error_p = np.zeros(iterations)
error_u1 = np.zeros(iterations)
error_u2 = np.zeros(iterations)
error_s1 = np.zeros(iterations)
error_s2 = np.zeros(iterations)

lump = 4
lump2 = 4

for k in range(iterations):
    phix_RT1_1, phiy_RT1_1 = pde.hdiv.assemble(MESH, space = 'RT1', matrix = 'M', order = 1)
    phix_RT1_2, phiy_RT1_2 = pde.hdiv.assemble(MESH, space = 'RT1', matrix = 'M', order = 2)
    phix_RT1_lump,  phiy_RT1_lump  = pde.hdiv.assemble(MESH, space = 'RT1', matrix = 'M', order = lump)
    phix_RT1_lump2, phiy_RT1_lump2 = pde.hdiv.assemble(MESH, space = 'RT1', matrix = 'M', order = lump2)
    
    divphi_RT1 = pde.hdiv.assemble(MESH, space = 'RT1', matrix = 'K', order = 2)
    
    phi_H1_lump2 = pde.h1.assemble(MESH, space = 'P1', matrix = 'M', order = lump2)
    phi_H1_2 = pde.h1.assemble(MESH, space = 'P1', matrix = 'M', order = 2)
    
    phi_L2 = pde.l2.assemble(MESH, space = 'P1d', matrix = 'M', order = 2)
    
    phix_NED1,   phiy_NED1   = pde.hcurl.assemble(MESH, space = 'N1', matrix = 'M', order = 2)
    phix_NED1_1, phiy_NED1_1 = pde.hcurl.assemble(MESH, space = 'N1', matrix = 'M', order = 1)
    phix_NED1_4, phiy_NED1_4 = pde.hcurl.assemble(MESH, space = 'N1', matrix = 'M', order = 4)
    
    divsigma1_eval = pde.int.evaluate(MESH, order = 4, coeff = divsigma1)
    divsigma2_eval = pde.int.evaluate(MESH, order = 4, coeff = divsigma2)
    
    D2_lump  = pde.int.assemble(MESH, order = lump)
    D2_lump2 = pde.int.assemble(MESH, order = lump2)
    
    D4 = pde.int.assemble(MESH, order = 4)
    D2 = pde.int.assemble(MESH, order = 2)
    D1 = pde.int.assemble(MESH, order = 1)
    D0 = pde.int.assemble(MESH, order = 0)
    
    rhs1 = phix_NED1_4@D4@divsigma1_eval.diagonal()
    rhs2 = phiy_NED1_4@D4@divsigma2_eval.diagonal()
    
    Mh_x = phix_RT1_lump@D2_lump@phix_RT1_lump.T
    Mh_y = phiy_RT1_lump@D2_lump@phiy_RT1_lump.T
    
    Mh = Mh_x + Mh_y
    
    C1_new = divphi_RT1@D2@phix_NED1.T
    C2_new = divphi_RT1@D2@phiy_NED1.T
    D = phi_L2@D2@phi_L2.T
    
    D_H1 = phi_H1_lump2@D2_lump2@phi_H1_lump2.T
    
    AS1 = phiy_RT1_lump2@D2_lump2@phi_H1_lump2.T
    AS2 = phix_RT1_lump2@D2_lump2@phi_H1_lump2.T
    
    from scipy.sparse import csc_matrix, hstack as sph, vstack as spv
    
    M = spv([sph([Mh,0*Mh]),
             sph([0*Mh,Mh])])
    
    T = spv([sph([Mh_x,0*Mh_x]),
             sph([0*Mh_y,Mh_y])])
    
    M = 1/(2*mu)*(M-lam/(2*mu+2*lam)*T)
    AS = spv([AS1,-AS2])
    
    C = spv([C1_new,
             C2_new])
    
    ndofs = 3*MESH.nt
    
    Zc = csc_matrix((C.shape[1],C.shape[1]))
    Zt = csc_matrix((C.shape[1],AS.shape[1]))
    Zs = csc_matrix((AS.shape[1],AS.shape[1]))
    
    
    SYS = spv([sph([M,C,AS]),
               sph([C.T,Zc,Zt]),
               sph([AS.T,Zt.T,Zs])]).tocsc()
    
    rhs = np.r_[np.zeros(M.shape[1]),
                rhs1+rhs2,
                np.zeros(AS.shape[1])]
    
    from scipy.sparse.linalg import spsolve
    res = spsolve(SYS, rhs)
    
    sh = res[0:M.shape[1]]
    uh = res[M.shape[1]:M.shape[1]+C.shape[1]]
    ph = res[M.shape[1]+C.shape[1]:]
    
    sh1 = sh[:2*MESH.NoEdges+2*MESH.nt]
    sh2 = sh[2*MESH.NoEdges+2*MESH.nt:]
    
    sh11 = phix_RT1_2.T@sh1
    sh12 = phiy_RT1_2.T@sh1
    sh21 = phix_RT1_2.T@sh2
    sh22 = phiy_RT1_2.T@sh2
    
    ph1_o2 = pde.int.evaluate(MESH, order = 2, coeff = p_test).diagonal()
    ph1 = phi_H1_2.T@ph
    
    ph_P1d = np.r_[p_test(MESH.p[:,0],MESH.p[:,1]),27*p_test(MESH.mp[:,0],MESH.mp[:,1])]
    # ph_P1d = np.r_[p_test(MESH.p[:,0],MESH.p[:,1])]
    
    uh1_P1d = pde.int.evaluate(MESH, order = 1, coeff = u1_test).diagonal()
    uh2_P1d = pde.int.evaluate(MESH, order = 1, coeff = u2_test).diagonal()
    
    uh1 = phix_NED1_1.T@uh
    uh2 = phiy_NED1_1.T@uh
    
    sh11_P1d = pde.int.evaluate(MESH, order = 2, coeff = sigma11).diagonal()
    sh12_P1d = pde.int.evaluate(MESH, order = 2, coeff = sigma12).diagonal()
    sh21_P1d = pde.int.evaluate(MESH, order = 2, coeff = sigma21).diagonal()
    sh22_P1d = pde.int.evaluate(MESH, order = 2, coeff = sigma22).diagonal()
    
    error_p[k] = np.sqrt((ph1-ph1_o2)@D2@(ph1-ph1_o2))
    
    error_u1[k] = np.sqrt((uh1-uh1_P1d)@D1@(uh1-uh1_P1d))
    error_u2[k] = np.sqrt((uh2-uh2_P1d)@D1@(uh2-uh2_P1d))
    
    
    error_s1[k] = np.sqrt((sh11-sh11_P1d)@D2@(sh11-sh11_P1d)) + np.sqrt((sh12-sh12_P1d)@D2@(sh12-sh12_P1d))
    error_s2[k] = np.sqrt((sh21-sh21_P1d)@D2@(sh21-sh21_P1d)) + np.sqrt((sh22-sh22_P1d)@D2@(sh22-sh22_P1d))
    
    # from scipy.linalg import interpolative
    # from scipy.sparse.linalg import LinearOperator
    # linSYS = LinearOperator(SYS.shape,matvec = lambda x: SYS@x, rmatvec = lambda x : SYS@x)
    # print(interpolative.estimate_rank(linSYS,eps = 1e-2),SYS.shape)
    
    # import numpy.linalg
    # print(numpy.linalg.matrix_rank(SYS.todense()),SYS.shape,numpy.linalg.cond(SYS.todense()))
    
    print("{:.2f}".format(pde.tools.condest(SYS)),SYS.shape)
    
    MESH.refinemesh()
    
error_u = error_u1 + error_u2
error_s = error_s1 + error_s2

print("\n")
    
rate_p = np.log2(error_p[1:-1]/error_p[2:])
print("Convergenge rates p : ",rate_p)
# print("Errors: ",error_p)

rate_u = np.log2(error_u[1:-1]/error_u[2:])
print("Convergenge rates u : ",rate_u)
# print("Errors: ",error_u)

rate_s = np.log2(error_s[1:-1]/error_s[2:])
print("Convergenge rates s : ",rate_s)
# print("Errors: ",error_s)

# fig = MESH.pdesurf_hybrid(dict(trig = 'P0', controls = 1), ph)
# fig.show()




################################################################################



################################################################################

################################################################################






# plt.spy(SYS,markersize=1)

# print(Mh.shape,C.shape,D.shape,AS1.shape,AS2.shape)
# print(C.shape,AS.shape,Zc.shape,Zt.shape,Zs.shape)
