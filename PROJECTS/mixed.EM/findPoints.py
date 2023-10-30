import numpy as np
import pde

# @profile
def getPoints(MESH):
    airL_index = MESH.getIndices2d(MESH.regions_1d,'airL')[0]
    airR_index = MESH.getIndices2d(MESH.regions_1d,'airR')[0]
    
    pointsL_index = np.unique(MESH.e[np.argwhere(MESH.e[:,2] == airL_index)[:,0],:2])
    pointsR_index = np.unique(MESH.e[np.argwhere(MESH.e[:,2] == airR_index)[:,0],:2])
    
    pointsL = MESH.p[pointsL_index,:]; pointsR = MESH.p[pointsR_index,:]
    
    indL = np.argsort(pointsL[:,0]**2) # es reicht, nach der ersten Koordinate zu sortieren!
    indR = np.argsort(pointsR[:,0]**2)
    
    pointsL_index_sorted = pointsL_index[indL]
    pointsR_index_sorted = pointsR_index[indR]
    
    edges0 = np.c_[pointsL_index_sorted[:-1],
                   pointsL_index_sorted[1:]]
                   
    edges1 = np.c_[pointsR_index_sorted[:-1],
                   pointsR_index_sorted[1:]]
    
    edges0 = np.sort(edges0)
    edges1 = np.sort(edges1)

    # edgecoord0 = np.zeros(edges0.shape[0],dtype=int)-1
    # edgecoord1 = np.zeros(edges1.shape[0],dtype=int)-1
    
    # tm3 = time.monotonic()
    # for i in range(edges0.shape[0]):
    #     v0 =  np.where(np.all(MESH.EdgesToVertices[:,:2]==edges0[i,:],axis=1))[0]
    #     v1 =  np.where(np.all(MESH.EdgesToVertices[:,:2]==edges1[i,:],axis=1))[0]
    #     edgecoord0[i] = v0
    #     edgecoord1[i] = v1
        
    
    edgecoord0 = pde.tools.ismember(edges0,MESH.EdgesToVertices[:,:2],'rows')[1]
    edgecoord1 = pde.tools.ismember(edges1,MESH.EdgesToVertices[:,:2],'rows')[1]    
        
    # print('loop took  ', time.monotonic()-tm)
    
    ident_points_gap = np.c_[pointsL_index_sorted,
                             pointsR_index_sorted]
    
    ident_edges_gap = np.c_[edgecoord0,
                            edgecoord1]
    
    return ident_points_gap, ident_edges_gap






# @profile
def getPointsNoEdges(MESH):
    airL_index = MESH.getIndices2d(MESH.regions_1d,'airL')[0]
    airR_index = MESH.getIndices2d(MESH.regions_1d,'airR')[0]
    
    pointsL_index = np.unique(MESH.e[np.argwhere(MESH.e[:,2] == airL_index)[:,0],:2])
    pointsR_index = np.unique(MESH.e[np.argwhere(MESH.e[:,2] == airR_index)[:,0],:2])
    
    pointsL = MESH.p[pointsL_index,:]; pointsR = MESH.p[pointsR_index,:]
    
    indL = np.argsort(pointsL[:,0]**2) # es reicht, nach der ersten Koordinate zu sortieren!
    indR = np.argsort(pointsR[:,0]**2)
    
    pointsL_index_sorted = pointsL_index[indL]
    pointsR_index_sorted = pointsR_index[indR]
    
    edges0 = np.c_[pointsL_index_sorted[:-1],
                   pointsL_index_sorted[1:]]
                   
    edges1 = np.c_[pointsR_index_sorted[:-1],
                   pointsR_index_sorted[1:]]
    
    edges0 = np.sort(edges0)
    edges1 = np.sort(edges1)
    
    ident_points_gap = np.c_[pointsL_index_sorted,
                             pointsR_index_sorted]
    
    return ident_points_gap


def makeIdentifications_nogap(MESH):

    a = MESH.identifications

    c0 = np.zeros(a.shape[0])
    c1 = np.zeros(a.shape[0])

    for i in range(a.shape[0]):
        point0 = MESH.p[a[i,0]-1]
        point1 = MESH.p[a[i,1]-1]

        c0[i] = point0[0]**2+point0[1]**2
        c1[i] = point1[0]**2+point1[1]**2

    ind0 = np.argsort(c0)

    aa = np.c_[a[ind0[:-1],0]-1,
                a[ind0[1: ],0]-1]

    edges0 = np.c_[a[ind0[:-1],0]-1,
                    a[ind0[1: ],0]-1]
    edges1 = np.c_[a[ind0[:-1],1]-1,
                    a[ind0[1: ],1]-1]

    edges0 = np.sort(edges0)
    edges1 = np.sort(edges1)

    edgecoord0 = np.zeros(edges0.shape[0],dtype=int)
    edgecoord1 = np.zeros(edges1.shape[0],dtype=int)

    for i in range(edges0.shape[0]): # modify this maybe later?
        edgecoord0[i] = np.where(np.all(MESH.EdgesToVertices[:,:2]==edges0[i,:],axis=1))[0][0]
        edgecoord1[i] = np.where(np.all(MESH.EdgesToVertices[:,:2]==edges1[i,:],axis=1))[0][0]
        
    identification = np.c_[np.r_[a[ind0,0]-1,MESH.np + edgecoord0],
                            np.r_[a[ind0,1]-1,MESH.np + edgecoord1]]
    ident_points = np.c_[a[ind0,0]-1,
                          a[ind0,1]-1]
    ident_edges = np.c_[edgecoord0,
                        edgecoord1]
    return ident_points, ident_edges


def makeIdentifications(MESH):

    a = MESH.identifications

    c0 = np.zeros(a.shape[0])
    c1 = np.zeros(a.shape[0])

    for i in range(a.shape[0]):
        point0 = MESH.p[a[i,0]-1]
        point1 = MESH.p[a[i,1]-1]

        c0[i] = point0[0]**2+point0[1]**2
        c1[i] = point1[0]**2+point1[1]**2
    
    r_sliding = 78.8354999*10**(-3)
    r_sliding2 = 79.03874999*10**(-3)
    
    i0 = np.where(np.abs(c0-r_sliding**2 )<1e-10)[0][0]
    i1 = np.where(np.abs(c0-r_sliding2**2)<1e-10)[0][0]
    
    
    ind0 = np.argsort(c0)
    
    jumps = np.r_[np.where(ind0==i0)[0][0],np.where(ind0==i1)[0][0]]
    
    aa = np.c_[a[ind0[:-1],0]-1,
               a[ind0[1: ],0]-1]

    edges0 = np.c_[a[ind0[:-1],0]-1,
                   a[ind0[1: ],0]-1]
    edges1 = np.c_[a[ind0[:-1],1]-1,
                   a[ind0[1: ],1]-1]

    edges0 = np.sort(edges0)
    edges1 = np.sort(edges1)

    # edgecoord0 = np.zeros(edges0.shape[0],dtype=int)-1
    # edgecoord1 = np.zeros(edges1.shape[0],dtype=int)-1
    
    # for i in range(edges0.shape[0]):
    #     v0 =  np.where(np.all(MESH.EdgesToVertices[:,:2]==edges0[i,:],axis=1))[0] #slow
    #     v1 =  np.where(np.all(MESH.EdgesToVertices[:,:2]==edges1[i,:],axis=1))[0] #slow
        
    #     if v0.size == 1:
    #         edgecoord0[i] = v0
    #         edgecoord1[i] = v1
    
    edgecoord0 = pde.tools.ismember(edges0,MESH.EdgesToVertices[:,:2],'rows')[1]
    edgecoord1 = pde.tools.ismember(edges1,MESH.EdgesToVertices[:,:2],'rows')[1]
    
    
    identification = np.c_[np.r_[a[ind0,0]-1,MESH.np + edgecoord0],
                           np.r_[a[ind0,1]-1,MESH.np + edgecoord1]]
    ident_points = np.c_[a[ind0,0]-1,
                         a[ind0,1]-1]
    ident_edges = np.c_[edgecoord0,
                        edgecoord1]
    
    # index = np.argwhere((ident_edges[:,0] == -1)*(ident_edges[:,1] == -1))[0]
    # if index.size ==1:
    #     ident_edges = np.delete(ident_edges, index, axis=0)

    return ident_points, ident_edges, jumps