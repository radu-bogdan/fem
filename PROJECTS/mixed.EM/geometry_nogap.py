from imports import *
from imports import pde,np

def load_m_j3():
    motor_npz = np.load('../meshes/data.npz', allow_pickle = True)
    m = motor_npz['m']; m_new = m
    j3 = motor_npz['j3']
    return m, j3
    

def loadGeometryStuff(level):
    open_file = open('mesh'+str(level)+'.pkl', "rb")
    MESH = dill.load(open_file)[0]
    open_file.close()
    return MESH

from findPoints import *

tm = time.monotonic()
ident_points_gap, ident_edges_gap = getPoints(MESH)
# ident_points_gap = getPointsNoEdges(MESH)
print('getPoints took  ', time.monotonic()-tm)

tm = time.monotonic()
ident_points, ident_edges, jumps = makeIdentifications(MESH)
print('makeIdentifications took  ', time.monotonic()-tm)


##########################################################################################
# Identifications
##########################################################################################

def getRS(k):
    
    if ORDER == 1:
        ident = ident_points
    if ORDER == 2:
        ident = np.r_[ident_points, MESH.np + ident_edges]
    
    i0 = ident[:,0]; i1 = ident[:,1]
    
    R_out, R_int = pde.h1.assembleR(MESH, space = poly, edges = 'left,right,airL,airR')
    # R_out, R_int = pde.h1.assembleR(MESH, space = poly, edges = 'stator_outer,left,right,airL,airR')
    R_L, R_LR = pde.h1.assembleR(MESH, space = poly, edges = 'left', listDOF = i1)
    R_R, R_RR = pde.h1.assembleR(MESH, space = poly, edges = 'right', listDOF = i0)
   
    # manual stuff: (removing the point in the three corners...)
    corners = np.r_[0,jumps,ident_points.shape[0]-1]
    ind1 = np.setdiff1d(np.r_[0:R_L.shape[0]], corners)
    R_L = R_L[ind1,:]
    
    corners = np.r_[0,jumps,ident_points.shape[0]-1]
    ind1 = np.setdiff1d(np.r_[0:R_R.shape[0]], corners)
    R_R = R_R[ind1,:]
    
    ident0 = np.roll(ident_points_gap[:,0], -k*rot_speed)
    ident1 = ident_points_gap[:,1]
    
    R_AL, R_ALR = pde.h1.assembleR(MESH, space = poly, edges = 'airL', listDOF = ident0)
    R_AR, R_ARR = pde.h1.assembleR(MESH, space = poly, edges = 'airR', listDOF = ident1)
        
    if k>0:
        R_AL[-k*rot_speed:,:] = -R_AL[-k*rot_speed:,:]
        
    if ORDER == 2:
        
        R_AL2, _ = pde.h1.assembleR(MESH, space = poly, edges = 'airL', listDOF = MESH.np + np.roll(ident_edges_gap[:,0], -k*rot_speed))
        R_AR2, _ = pde.h1.assembleR(MESH, space = poly, edges = 'airR', listDOF = MESH.np + ident_edges_gap[:,1])
        
        if k>0:
            R_AL2[-k*rot_speed:,:] = -R_AL2[-k*rot_speed:,:] # old
            
        from scipy.sparse import bmat
        R_AL =  bmat([[R_AL], [R_AL2]])
        R_AR =  bmat([[R_AR], [R_AR2]])
        
    
    from scipy.sparse import bmat
    RS =  bmat([[R_int], [R_L-R_R], [R_AL+R_AR]])