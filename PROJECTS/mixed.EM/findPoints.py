import numpy as np
import pde
from scipy.sparse import bmat

# @profile

def getPoints(MESH):
    airL_index = pde.tools.getIndices(MESH.regions_1d,'airL')[0]
    airR_index = pde.tools.getIndices(MESH.regions_1d,'airR')[0]
    
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
    
    MESH.ident_points_gap = ident_points_gap
    MESH.ident_edges_gap = ident_edges_gap
    # return MESH


# @profile
def getPointsNoEdges(MESH):
    airL_index = pde.tools.getIndices(MESH.regions_1d,'airL')[0]
    airR_index = pde.tools.getIndices(MESH.regions_1d,'airR')[0]
    
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
    MESH.ident_points_gap = ident_points_gap
    # return ident_points_gap


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
    MESH.ident_points = ident_points
    MESH.ident_edges = ident_edges
    # return ident_points, ident_edges


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
    
    MESH.ident_points = ident_points
    MESH.ident_edges = ident_edges
    MESH.jumps = jumps
    
    # return ident_points, ident_edges, jumps
    
    
def getRS_H1(MESH,ORDER,poly,k,rot_speed):
    if ORDER == 1:
        ident = MESH.ident_points
    if ORDER == 2:
        ident = np.r_[MESH.ident_points, MESH.np + MESH.ident_edges]
    
    i0 = ident[:,0]; i1 = ident[:,1]
    
    R_out, R_int = pde.h1.assembleR(MESH, space = poly, edges = 'stator_outer,left,right,airL,airR')
    R_L, R_LR = pde.h1.assembleR(MESH, space = poly, edges = 'left', listDOF = i1)
    R_R, R_RR = pde.h1.assembleR(MESH, space = poly, edges = 'right', listDOF = i0)
   
    # manual stuff: (removing the point in the three corners...)
    corners = np.r_[0,MESH.jumps,MESH.ident_points.shape[0]-1]
    ind1 = np.setdiff1d(np.r_[0:R_L.shape[0]], corners)
    R_L = R_L[ind1,:]
    
    corners = np.r_[0,MESH.jumps,MESH.ident_points.shape[0]-1]
    ind1 = np.setdiff1d(np.r_[0:R_R.shape[0]], corners)
    R_R = R_R[ind1,:]
    
    
    ident0 = np.roll(MESH.ident_points_gap[:,0], -k*rot_speed)
    ident1 = MESH.ident_points_gap[:,1]
    
    R_AL, R_ALR = pde.h1.assembleR(MESH, space = poly, edges = 'airL', listDOF = ident0)
    R_AR, R_ARR = pde.h1.assembleR(MESH, space = poly, edges = 'airR', listDOF = ident1)
        
    if k>0:
        R_AL[-k*rot_speed:,:] = -R_AL[-k*rot_speed:,:]
        
    if ORDER == 2:
        
        R_AL2, _ = pde.h1.assembleR(MESH, space = poly, edges = 'airL', listDOF = MESH.np + np.roll(MESH.ident_edges_gap[:,0], -k*rot_speed))
        R_AR2, _ = pde.h1.assembleR(MESH, space = poly, edges = 'airR', listDOF = MESH.np + MESH.ident_edges_gap[:,1])
        
        if k>0:
            R_AL2[-k*rot_speed:,:] = -R_AL2[-k*rot_speed:,:] # old
            
        
        R_AL =  bmat([[R_AL], [R_AL2]])
        R_AR =  bmat([[R_AR], [R_AR2]])
        
    
    RS =  bmat([[R_int], [R_L-R_R], [R_AL+R_AR]])
    
    return RS


def getRS_Hcurl(MESH,ORDER,poly,k,rot_speed):
    R_out, R_int = pde.hcurl.assembleR(MESH, space = poly, edges = 'stator_outer,left,right,airR,airL')
    
    
    EdgeDirection = (-1)*np.r_[np.sign(MESH.ident_points[1:MESH.jumps[1],:]-MESH.ident_points[:MESH.jumps[0],:]),
                               np.sign(MESH.ident_points[MESH.jumps[1]+1:,:]-MESH.ident_points[MESH.jumps[1]:-1,:])]
    
    
    EdgeDirectionGap = (-1)*np.sign(MESH.ident_points_gap[1:,:].astype(int)-MESH.ident_points_gap[:-1,:].astype(int))
    
    if ORDER == 1:
        ind_per_0 = MESH.ident_edges[:,0]
        ind_per_1 = MESH.ident_edges[:,1]
        
        ident_edges_gap_0_rolled = np.roll(MESH.ident_edges_gap[:,0], -k*rot_speed)
        
        EdgeDirectionGap_rolled = np.roll(EdgeDirectionGap[:,0], -k*rot_speed)
        EdgeDirectionGap_1 = EdgeDirectionGap[:,1]
        
        ind_gap_0 = ident_edges_gap_0_rolled
        ind_gap_1 = MESH.ident_edges_gap[:,1]
        
    if ORDER > 1:
        
        ind_per_0 = np.c_[2*MESH.ident_edges[:,0]   -1/2*(EdgeDirection[:,0]-1),
                          2*MESH.ident_edges[:,0]+1 +1/2*(EdgeDirection[:,0]-1)].ravel()
        
        ind_per_1 = np.c_[2*MESH.ident_edges[:,1]   -1/2*(EdgeDirection[:,1]-1),
                          2*MESH.ident_edges[:,1]+1 +1/2*(EdgeDirection[:,1]-1)].ravel()
        
        
        ident_edges_gap_0_rolled = np.roll(MESH.ident_edges_gap[:,0], -k*rot_speed)
        EdgeDirectionGap_rolled = np.roll(EdgeDirectionGap[:,0], -k*rot_speed)
        
        EdgeDirectionGap_1 = np.repeat(EdgeDirectionGap[:,1],2)
        
        
        ind_gap_0 = np.c_[2*ident_edges_gap_0_rolled   -1/2*(EdgeDirectionGap_rolled-1),
                          2*ident_edges_gap_0_rolled+1 +1/2*(EdgeDirectionGap_rolled-1)].ravel()
        
        ind_gap_1 = np.c_[2*MESH.ident_edges_gap[:,1]   -1/2*(EdgeDirectionGap[:,1]-1),
                          2*MESH.ident_edges_gap[:,1]+1 +1/2*(EdgeDirectionGap[:,1]-1)].ravel()
        
        EdgeDirectionGap_rolled = np.roll(np.repeat(EdgeDirectionGap[:,0],2), -2*k*rot_speed)
        
        
    R_L, R_LR = pde.hcurl.assembleR(MESH, space = poly, edges = 'left', listDOF = ind_per_0)
    R_R, R_RR = pde.hcurl.assembleR(MESH, space = poly, edges = 'right', listDOF = ind_per_1)
    
    R_AL, R_ALR = pde.hcurl.assembleR(MESH, space = poly, edges = 'airL', listDOF = ind_gap_0); #R_AL.data = EdgeDirectionGap_rolled[R_AL.indices]
    R_AR, R_ARR = pde.hcurl.assembleR(MESH, space = poly, edges = 'airR', listDOF = ind_gap_1); #R_AR.data = EdgeDirectionGap_1[R_AR.indices]
    
    if k>0:
        if ORDER == 1:
            R_AL[-k*rot_speed:,:] = -R_AL[-k*rot_speed:,:]
            
        if ORDER > 1:
            R_AL[-2*k*rot_speed:,:] = -R_AL[-2*k*rot_speed:,:]
    
    RS =  bmat([[R_int], [R_L-R_R], [R_AL+R_AR]])
    
    return RS
    
def getRS_H1_nonzero(MESH,ORDER,poly,k,rot_speed):
    if ORDER == 1:
        ident = MESH.ident_points
    if ORDER == 2:
        ident = np.r_[MESH.ident_points, MESH.np + MESH.ident_edges]
    
    i0 = ident[:,0]; i1 = ident[:,1]
    
    R_out, R_int = pde.h1.assembleR(MESH, space = poly, edges = 'left,right,airL,airR')
    R_L, R_LR = pde.h1.assembleR(MESH, space = poly, edges = 'left', listDOF = i1)
    R_R, R_RR = pde.h1.assembleR(MESH, space = poly, edges = 'right', listDOF = i0)
   
    # manual stuff: (removing the point in the three corners...)
    corners = np.r_[0,MESH.jumps]
    # corners = np.r_[0,MESH.jumps,MESH.ident_points.shape[0]-1]
    ind1 = np.setdiff1d(np.r_[0:R_L.shape[0]], corners)
    R_L = R_L[ind1,:]
    
    corners = np.r_[0,MESH.jumps]
    # corners = np.r_[0,MESH.jumps,MESH.ident_points.shape[0]-1]
    ind1 = np.setdiff1d(np.r_[0:R_R.shape[0]], corners)
    R_R = R_R[ind1,:]
    
    
    
    ident0 = np.roll(MESH.ident_points_gap[:,0], -k*rot_speed)
    ident1 = MESH.ident_points_gap[:,1]
    
    R_AL, R_ALR = pde.h1.assembleR(MESH, space = poly, edges = 'airL', listDOF = ident0)
    R_AR, R_ARR = pde.h1.assembleR(MESH, space = poly, edges = 'airR', listDOF = ident1)
        
    if k>0:
        R_AL[-k*rot_speed:,:] = -R_AL[-k*rot_speed:,:]
        
    if ORDER == 2:
        
        R_AL2, _ = pde.h1.assembleR(MESH, space = poly, edges = 'airL', listDOF = MESH.np + np.roll(MESH.ident_edges_gap[:,0], -k*rot_speed))
        R_AR2, _ = pde.h1.assembleR(MESH, space = poly, edges = 'airR', listDOF = MESH.np + MESH.ident_edges_gap[:,1])
        
        if k>0:
            R_AL2[-k*rot_speed:,:] = -R_AL2[-k*rot_speed:,:] # old
            
        
        R_AL =  bmat([[R_AL], [R_AL2]])
        R_AR =  bmat([[R_AR], [R_AR2]])
        
    
    RS =  bmat([[R_int], [R_L-R_R], [R_AL+R_AR]])
    
    return RS