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
iterations = 6

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

error_p = np.zeros(iterations);
error_u1 = np.zeros(iterations);
error_u2 = np.zeros(iterations);
error_s1 = np.zeros(iterations);
error_s2 = np.zeros(iterations);

lump = '2l'
lump2 = 1

for k in range(iterations):
    qMhx,qMhy = pde.hdiv.assemble(MESH, space = 'RT1', matrix = 'M', order = lump)
    qMx,qMy = pde.hdiv.assemble(MESH, space = 'RT1', matrix = 'M', order = 4)
    qK = pde.hdiv.assemble(MESH, space = 'RT1', matrix = 'K', order = 2)
        
    qD4 = pde.h1.assemble(MESH, space = 'P1', matrix = 'M', order = lump2)
    
    # qD4 = pde.l2.assemble(MESH, space = 'P1d', matrix = 'M', order = 4)
    qD2 = pde.l2.assemble(MESH, space = 'P1d', matrix = 'M', order = 2)
    qD2l = pde.l2.assemble(MESH, space = 'P1d', matrix = 'M', order = lump2)
    
    divsigma1_eval = pde.int.evaluate(MESH, order = 2, coeff = divsigma1)
    divsigma2_eval = pde.int.evaluate(MESH, order = 2, coeff = divsigma2)
    
    D2l = pde.int.assemble(MESH, order = lump)
    D4 = pde.int.assemble(MESH, order = 4)
    D2 = pde.int.assemble(MESH, order = 2)
    D1 = pde.int.assemble(MESH, order = 1)
    D0 = pde.int.assemble(MESH, order = 0)
    
    rhs1 = qD2@D2@divsigma1_eval.diagonal()
    rhs2 = qD2@D2@divsigma2_eval.diagonal()
    
    Mh_x = qMhx@D2l@qMhx.T
    Mh_y = qMhy@D2l@qMhy.T
    
    Mh = Mh_x + Mh_y
    C = qK@D2@qD2.T
    D = qD2@D2@qD2.T
    
    D_H1 = qD4@D2l@qD4.T
    
    # AS1 = qMhy@D2l@qD2l.T
    # AS2 = qMhx@D2l@qD2l.T
    
    AS1 = qMhy@D2l@qD4.T
    AS2 = qMhx@D2l@qD4.T
    
    from scipy.sparse import hstack as sph
    from scipy.sparse import vstack as spv
    from scipy.sparse import csc_matrix
    
    M = spv([sph([Mh,0*Mh]),
             sph([0*Mh,Mh])])
    
    T = spv([sph([Mh_x,0*Mh_x]),
             sph([0*Mh_y,Mh_y])])
    
    M = 1/(2*mu)*(M-lam/(2*mu+2*lam)*T)
    AS = spv([AS1,-AS2])
    C = spv([sph([C,0*C]),
             sph([0*C,C])])
    
    
    ndofs = 3*MESH.nt
    ndofs2 = MESH.np
    
    Zc = csc_matrix((2*ndofs,2*ndofs))
    Zs = csc_matrix((ndofs2,ndofs2))
    Zt = csc_matrix((2*ndofs,ndofs2))
    
    
    SYS = spv([sph([M,C,AS]),
               sph([C.T,Zc,Zt]),
               sph([AS.T,Zt.T,Zs])]).tocsc()
    
    rhs = np.r_[np.zeros(M.shape[1]),
                rhs1,
                rhs2,
                np.zeros(AS.shape[1])]
    
    from scipy.sparse.linalg import spsolve
    res = spsolve(SYS, rhs)
    
    sh = res[0:M.shape[1]]
    uh = res[M.shape[1]:M.shape[1]+C.shape[1]]
    ph = res[M.shape[1]+C.shape[1]:]
    
    uh1 = uh[0:ndofs]
    uh2 = uh[ndofs:]
    
    sh1 = sh[0:2*MESH.NoEdges+2*MESH.nt]
    sh2 = sh[2*MESH.NoEdges+2*MESH.nt:]
    
    # ph_P1d = pde.int.evaluate(MESH, order = 1, coeff = p_test).diagonal()
    ph_P1d = p_test(MESH.p[:,0],MESH.p[:,1])
    uh1_P1d = pde.int.evaluate(MESH, order = 1, coeff = u1_test).diagonal()
    uh2_P1d = pde.int.evaluate(MESH, order = 1, coeff = u2_test).diagonal()
    
    # sh1_BDM1 = pde.hdiv.interp(MESH, space = 'BDM1', order = 5, f = lambda x,y : np.c_[sigma11(x,y),sigma12(x,y)])
    # sh2_BDM1 = pde.hdiv.interp(MESH, space = 'BDM1', order = 5, f = lambda x,y : np.c_[sigma21(x,y),sigma22(x,y)])
    
    error_p[k] = np.sqrt((ph-ph_P1d)@D_H1@(ph-ph_P1d))
    error_u1[k] = np.sqrt((uh1-uh1_P1d)@D1@(uh1-uh1_P1d))
    error_u2[k] = np.sqrt((uh2-uh2_P1d)@D1@(uh2-uh2_P1d))
    
    
    
    # error_s1[k] = np.sqrt((sh1-sh1_BDM1)@Mh@(sh1-sh1_BDM1))
    # error_s2[k] = np.sqrt((sh2-sh2_BDM1)@Mh@(sh2-sh2_BDM1))
    
    # from scipy.linalg import interpolative
    # from scipy.sparse.linalg import LinearOperator
    # linSYS = LinearOperator(SYS.shape,matvec = lambda x: SYS@x, rmatvec = lambda x : SYS@x)
    # print(interpolative.estimate_rank(linSYS,eps = 1e-2),SYS.shape)
    
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
