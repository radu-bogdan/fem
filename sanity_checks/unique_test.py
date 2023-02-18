from __future__ import annotationsimport syssys.path.insert(0,'..') # adds parent directorysys.path.insert(0,'../CEM') # adds parent directoryimport numpy as npimport numba as nbimport pandas as pdimport gmshimport pdeimport scipy.sparse as spsimport scipy.sparse.linalgimport timeimport geometriesfrom sksparse.cholmod import choleskyimport plotly.io as piopio.renderers.default = 'browser'from matplotlib.pyplot import spyimport pde.toolsnp.set_printoptions(threshold = np.inf)np.set_printoptions(linewidth = np.inf)np.set_printoptions(precision = 8)gmsh.initialize()gmsh.model.add("Capacitor plates")geometries.unitSquare()gmsh.option.setNumber("Mesh.Algorithm", 2)gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)# gmsh.option.setNumber("Mesh.MeshSizeMax", 0.8)# gmsh.option.setNumber("Mesh.MeshSizeMin", 0.8)p,e,t,q = pde.petq_generate()tm = time.time()MESH = pde.mesh(p,e,t,q)elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))# MESH.refinemesh(); print(MESH.p.shape)# MESH.refinemesh(); print(MESH.p.shape)# MESH.refinemesh(); print(MESH.p.shape)MESH.refinemesh(); print(MESH.p.shape)MESH.refinemesh(); print(MESH.p.shape)MESH.refinemesh(); print(MESH.p.shape)MESH.refinemesh(); print(MESH.p.shape)MESH.refinemesh(); print(MESH.p.shape)MESH.makeRest()# tm = time.time()# p,e,t = MESH.refinemesh()# MESH = pde.mesh(p,e,t,q)# elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))# tm = time.time()# p,e,t = MESH.refinemesh()# MESH = pde.mesh(p,e,t,q)# elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))# tm = time.time()# p,e,t = MESH.refinemesh()# MESH = pde.mesh(p,e,t,q)# elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))# tm = time.time()# p,e,t = MESH.refinemesh()# MESH = pde.mesh(p,e,t,q)# elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))# tm = time.time()# p,e,t = MESH.refinemesh()# MESH = pde.mesh(p,e,t,q)# elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))# tm = time.time()# p,e,t = MESH.refinemesh()# MESH = pde.mesh(p,e,t,q)# elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))# tm = time.time()# p,e,t = MESH.refinemesh()# MESH = pde.mesh(p,e,t,q)# elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))edges_trigs = np.r_[np.c_[t[:,1],t[:,2]],                    np.c_[t[:,2],t[:,0]],                    np.c_[t[:,0],t[:,1]]]edges = np.sort(edges_trigs)print('lets go:')@nb.njit(parallel=True)def nb_argsort(finaltable):    indexTable = np.empty_like(finaltable)    for j in nb.prange(indexTable.shape[1]):        indexTable[:, j] = np.argsort(finaltable[:, j])    return indexTable# def do():# c, d = np.unique(edges,axis=0, return_inverse=True)# tm = time.time()# a,b,_ = pde.tools.nb_unique(edges, axis=0)# elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))# tm = time.time()# b = nb_argsort(edges)# elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))tm = time.time()c,d = pde.tools.unique_rows(edges, return_inverse=True)elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))tm = time.time()kk = np.lexsort(edges.T)elapsed = time.time()-tm; print('lex Took  {:4.8f} seconds.'.format(elapsed))tm = time.time()kkk = np.argsort(edges.T)elapsed = time.time()-tm; print('arg Took  {:4.8f} seconds.'.format(elapsed))# tm = time.time()# e = np.lexsort(edges.T)# f = np.argsort(edges.T)# e = pd.unique(edges)# elapsed = time.time()-tm; print('Took  {:4.8f} seconds.'.format(elapsed))# print(edges.T,'\n')# print(a.T,'\n')# print(b.T,'\n')# print(e.T,'\n')# return a,b,c,d,e    # a,b,c,d,e = do()nt = t.shape[0]nq = q.shape[0]EdgesToVertices, je = pde.unique_rows(edges, return_inverse = True)NoEdges = EdgesToVertices.shape[0]TriangleToEdges = je[0:3*nt].reshape(nt,3, order='F').astype(np.int64)QuadToEdges = je[3*nt:].reshape(nq,4, order='F')loc_trig,index_trig = MESH._mesh__ismember(MESH.TriangleToEdges,MESH.Boundary_Edges)loc_quad,index_quad = MESH._mesh__ismember(MESH.QuadToEdges,MESH.Boundary_Edges)indices_boundary = np.r_[index_trig,index_quad]direction_boundary = np.r_[MESH.EdgeDirectionTrig[loc_trig],MESH.EdgeDirectionQuad[loc_quad]]b = np.argsort(indices_boundary)Boundary_EdgeOrientation = direction_boundary[b]Boundary_EdgeOrientation2 = Boundary_EdgeOrientation.copy()_,ii,iii = np.intersect1d(MESH.TriangleToEdges.flatten(),MESH.Boundary_Edges,return_indices = True)Boundary_EdgeOrientation2[iii] = MESH.EdgeDirectionTrig.flatten()[ii]a = np.linalg.norm(MESH.Boundary_EdgeOrientation-MESH.Boundary_EdgeOrientation2)print(a)