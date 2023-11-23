from imports import *
from imports import pde,np
from scipy.sparse import bmat

from findPoints import *

tm = time.monotonic()
ident_points_gap, ident_edges_gap = getPoints(MESH)
# ident_points_gap = getPointsNoEdges(MESH)
print('getPoints took  ', time.monotonic()-tm)

tm = time.monotonic()
ident_points, ident_edges, jumps = makeIdentifications(MESH)
print('makeIdentifications took  ', time.monotonic()-tm)

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

##########################################################################################
# Identifications
##########################################################################################

