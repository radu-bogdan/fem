#!/usr/bin/python --relpath_append ../# @profile# def kek():import syssys.path.insert(0,'..') # adds parent directoryimport numpy as npimport gmshimport pdeimport scipy.sparse as spsimport scipy.sparse.linalgimport timeimport geometriesfrom sksparse.cholmod import choleskyfrom matplotlib.pyplot import spyimport plotly.io as piopio.renderers.default = 'browser'# pio.renderers.default = 'svg'np.set_printoptions(threshold = np.inf)np.set_printoptions(linewidth = np.inf)np.set_printoptions(precision = 8)# p,e,t,q = pde.petq_from_gmsh(filename = 'unit_square.geo',hmax = 0.3)d = 3l = 10gmsh.initialize()gmsh.model.add("Capacitor plates")geometries.capacitorPlates(a = 20,b = 20,c = 0.5,d = d,l = l)gmsh.option.setNumber("Mesh.Algorithm", 2)gmsh.option.setNumber("Mesh.MeshSizeMax", 1)gmsh.option.setNumber("Mesh.MeshSizeMin", 1)# gmsh.fltk.run()tm = time.time()p,e,t,q = pde.petq_generate()MESH = pde.mesh(p,e,t,q)MESH.refinemesh()MESH.refinemesh()# MESH.refinemesh()# MESH.refinemesh()# MESH.refinemesh()# MESH.refinemesh()print('Generating mesh and refining took {:4.8f} seconds.'.format(time.time()-tm))# MESH.makeRest()# TODO:  MESH = pde.refinemesh(p,e,t,q)# BASIS = pde.basis()# LISTS = pde.lists(MESH)f = lambda x,y : 0*np.sin(np.pi*x)*np.sin(np.pi*y)g1 = lambda x,y : -1+0*xg2 = lambda x,y :  1+0*x###############################################################################tm = time.time()Kx,Ky = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 0)D0 = pde.int.assemble(MESH, order = 0)Kxx = Kx@D0@Kx.T; Kyy = Ky@D0@Ky.Twalls = np.r_[5,6,7,8,9,10,11,12]left_block = np.r_[5,6,7,8]right_block = np.r_[9,10,11,12]Mb = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = Kxx.shape, order = 2)Db0 = pde.int.assembleB(MESH, order = 2)D_left_block = pde.int.evaluateB(MESH, order = 2, edges = left_block)D_right_block = pde.int.evaluateB(MESH, order = 2, edges = right_block)D_walls = pde.int.evaluateB(MESH, order = 2, edges = walls)B_full = Mb@Db0@Mb.TB_left_block = Mb@Db0@D_left_block@Mb.TB_right_block = Mb@Db0@D_right_block@Mb.TB_walls = Mb@Db0@D_walls@Mb.TD_g1 = pde.int.evaluateB(MESH, order = 2, edges = left_block, coeff = g1)D_g2 = pde.int.evaluateB(MESH, order = 2, edges = right_block, coeff = g2)B_g = Mb@Db0@(D_left_block@D_g1.diagonal() + D_right_block@D_g2.diagonal())gamma = 10**8A = Kxx + Kyy + gamma*B_wallsb = gamma*B_gprint('Assembling took {:4.8f} seconds.'.format(time.time()-tm))###############################################################################tm = time.time()cholA = cholesky(A)phi = cholA(b)# phi = sps.linalg.spsolve(A,b)oneM = np.ones(Kxx.shape[0])Q = gamma*(oneM@B_left_block@(phi+1)) #+ oneM@B_right_block@(phi-1)# Q = gamma*(oneM@B_right_block@(phi-1))print(str(Q/2), ' soll: ', str(l/d))elapsed = time.time()-tmprint('Solving took {:4.8f} seconds.'.format(elapsed))fig = MESH.pdesurf_hybrid(dict(trig = 'P1', controls = 1), phi)fig.show()# kek()