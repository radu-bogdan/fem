#!/usr/bin/python --relpath_append ../import syssys.path.insert(0,'..') # adds parent directoryimport numpy as npimport gmshimport pdeimport scipy.sparse as spsimport scipy.sparse.linalgimport timeimport geometriesimport plotly.io as piopio.renderers.default = 'browser'# pio.renderers.default = 'svg'from matplotlib.pyplot import spynp.set_printoptions(threshold = np.inf)np.set_printoptions(linewidth = np.inf)np.set_printoptions(precision = 8)# p,e,t,q = pde.petq_from_gmsh(filename = 'unit_square.geo',hmax = 0.3)# gmsh.initialize()d = 3l = 10gmsh.initialize()gmsh.model.add("Capacitor plates")# geometries.unitSquare()geometries.geometryP2()gmsh.option.setNumber("Mesh.Algorithm", 2)gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)# gmsh.fltk.run()p,e,t,q = pde.petq_generate()MESH = pde.mesh(p,e,t,q)MESH.makeFemLists()# TODO:  MESH = pde.refinemesh(p,e,t,q)BASIS = pde.basis()LISTS = pde.lists(MESH)f = lambda x,y : 0*np.sin(np.pi*x)*np.sin(np.pi*y)g1 = lambda x,y : -1+0*xg2 = lambda x,y :  1+0*xnu1 = lambda x,y : 1/1000 + 0*x +0*ynu2 = lambda x,y : 1 + 0*x +0*yKxx1,Kyy1,Kxy1,Kyx1 = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'K', coeff = nu1, regions = np.r_[2,3]))Kxx2,Kyy2,Kxy2,Kyx2 = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'K', coeff = nu2, regions = np.r_[1,4,5,6,7,8]))Kxx = Kxx1 + Kxx2; Kyy = Kyy1 + Kyy2Kxy = Kxy1 + Kxy2; Kyx = Kyx1 + Kyx2M = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'M'))Cx,Cy = pde.assemble.h1(MESH, BASIS, LISTS, dict(space = 'P1', matrix = 'C'))BM, DM = pde.h1.assemble(MESH, space = 'P1', matrix = 'M')M2 = BM@DM@BM.Ttm = time.time()BKx,BKy,DK = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 0)D1 = pde.h1.assembleD(MESH, order = 0, coeff = nu1, regions = np.r_[2,3])D2 = pde.h1.assembleD(MESH, order = 0, coeff = nu2, regions = np.r_[1,4,5,6,7,8])elapsed = time.time()-tmprint('Assembling P11 took ' + str(elapsed)[0:6] + ' seconds.')BD,DD = pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = 0)D = BD@DD@BD.TMAT = pde.assemble.hdiv(MESH, BASIS, LISTS, space = 'BDM1-BDM1'); DC = MAT['BDM1-BDM1']['D'];# BKx,BKy,DK = pde.h1.assemble(MESH,BASIS,LISTS, dict(space = 'P1', matrix = 'K'))tm = time.time()Kxx2 = BKx@DK@(D1+D2)@BKx.TKyy2 = BKy@DK@(D1+D2)@BKy.TKxy2 = BKx@DK@(D1+D2)@BKy.TKyx2 = BKy@DK@(D1+D2)@BKx.Telapsed = time.time()-tmprint('2Solving took ' + str(elapsed)[0:6] + ' seconds.')BKx,BKy,DK = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 1)BD,DD = pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = 1)Cx2 = BD@DD@BKx.TCy2 = BD@DD@BKy.TBB, DB = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = M.shape, order = 2)B2 = BB@DB@BB.TB = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P1', size = M.shape[0]))walls = np.r_[5,6,7,8,9,10,11,12]left_block = np.r_[5,6,7,8]right_block = np.r_[9,10,11,12]B_left_block = pde.assemble.h1b(MESH, BASIS, LISTS, dict(space = 'P1', size = M.shape[0], edges = left_block))B_right_block = pde.assemble.h1b(MESH, BASIS, LISTS, dict(space = 'P1', size = M.shape[0], edges = right_block))B_walls = pde.assemble.h1b(MESH, BASIS, LISTS, dict(space = 'P1', size = M.shape[0], edges = walls))D_left_block = pde.h1.assembleDB(MESH, order = 2, edges = left_block)D_right_block = pde.h1.assembleDB(MESH, order = 2, edges = right_block)D_walls = pde.h1.assembleDB(MESH, order = 2,  edges = walls)B2_left_block = BB@D_left_block@BB.TB2_right_block = BB@D_right_block@BB.TB2_walls = BB@D_walls@BB.Tf = lambda x,y : np.sin(np.pi*x)*np.sin(np.pi*y)M_f = pde.projections.assemH1(MESH, BASIS, LISTS, dict(trig = 'P1'), f)# BM, DM = pde.h1.assemble(MESH, space = 'P1', matrix = 'M')D1 = pde.h1.assembleD(MESH, order = 2, coeff = f)M_f2 = BM@DM@D1.diagonal()D_g1 = pde.h1.assembleDB(MESH, order = 2, edges = left_block)D_g2 = pde.h1.assembleDB(MESH, order = 2, edges = right_block)B_g  = pde.projections.assem_H1_b(MESH, BASIS, LISTS, dict(space = 'P1', order = 2, edges = left_block, size = M.shape[0]), g1)B_g += pde.projections.assem_H1_b(MESH, BASIS, LISTS, dict(space = 'P1', order = 2, edges = right_block, size = M.shape[0]), g2)B_g2 = BB@(D_left_block@D_g1.diagonal() + D_right_block@D_g2.diagonal())# tm = time.time()# MM = BM@DM@BM.T# elapsed = time.time()-tm# print('2Solving took ' + str(elapsed)[0:6] + ' seconds.')# B2, D2 = pde.assemb.h1.h1(MESH,BASIS,LISTS,dict(space = 'P1', matrix = 'M'))# tm = time.time()# M2 = B2@D2@B2.T# elapsed = time.time()-tm# print('3Solving took ' + str(elapsed)[0:6] + ' seconds.')# B3, D3 = pde.assemb.h1.h1(MESH,BASIS,LISTS,dict(space = 'P1', matrix = 'M3'))# # BB = B3@(np.sqrt(D3))# tm = time.time()# M3 = B3@D3@B3.T# # M3 = BB@BB.T# elapsed = time.time()-tm# print('4Solving took ' + str(elapsed)[0:6] + ' seconds.')# print(sps.linalg.norm(M-M2))print(sps.linalg.norm(Kxx-Kxx2))print(sps.linalg.norm(Kyy-Kyy2))print(sps.linalg.norm(Kxy-Kyx2))print(sps.linalg.norm(Kyx-Kxy2))print(sps.linalg.norm(D-DC))print(sps.linalg.norm(M-M2))print(sps.linalg.norm(Cx-Cx2))print(sps.linalg.norm(Cy-Cy2))print(sps.linalg.norm(B-B2))print(sps.linalg.norm(B_left_block-B2_left_block))print(sps.linalg.norm(B_right_block-B2_right_block))print(sps.linalg.norm(B_walls-B2_walls))print(np.linalg.norm(M_f-M_f2))print(np.linalg.norm(B_g-B_g2))# print(sps.linalg.norm(M0-M1))# print(sps.linalg.norm(M0-M2))# print(sps.linalg.norm(M0-M3))