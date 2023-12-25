
from .. import quadrature
import numpy as np

def spaceInfo(MESH,space):
    
    LISTS = MESH.FEMLISTS
    
    ###########################################################################
    if space == 'P1':
        
        LISTS['P1'] = {}
        LISTS['P1']['TET'] = {}
        
        LISTS['P1']['TET']['sizeM'] = MESH.np
        LISTS['P1']['TET']['qp_we_M'] = quadrature.keast(order = 2)
        LISTS['P1']['TET']['qp_we_Mh'] = quadrature.keast(order = 1)
        LISTS['P1']['TET']['qp_we_K'] = quadrature.keast(order = 0)
        
        LISTS['P1']['TET']['phi'] = {}
        LISTS['P1']['TET']['phi'][0] = lambda x,y,z: 1-x-y-z
        LISTS['P1']['TET']['phi'][1] = lambda x,y,z: x
        LISTS['P1']['TET']['phi'][2] = lambda x,y,z: y
        LISTS['P1']['TET']['phi'][3] = lambda x,y,z: z
        
        LISTS['P1']['TET']['dphi'] = {}
        LISTS['P1']['TET']['dphi'][0] = lambda x,y,z: np.r_[-1,-1,-1]
        LISTS['P1']['TET']['dphi'][1] = lambda x,y,z: np.r_[ 1, 0, 0]
        LISTS['P1']['TET']['dphi'][2] = lambda x,y,z: np.r_[ 0, 1 ,0]
        LISTS['P1']['TET']['dphi'][3] = lambda x,y,z: np.r_[ 0, 0, 1]
        
        LISTS['P1']['B'] = {}
        LISTS['P1']['B']['phi'] = {}
        LISTS['P1']['B']['phi'][0] = lambda x,y: 1-x-y # is this correct? prolly ...
        LISTS['P1']['B']['phi'][1] = lambda x,y: x
        LISTS['P1']['B']['phi'][2] = lambda x,y: y
        LISTS['P1']['B']['qp_we_B'] = quadrature.dunavant(order = 2)
        
        LISTS['P1']['TET']['LIST_DOF'] = MESH.t[:,:4]
        LISTS['P1']['B']['LIST_DOF'] = MESH.f[:,:3]
    ###########################################################################
    