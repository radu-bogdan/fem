
from .. import quadrature
import numpy as np

def spaceInfo(MESH,space):
    
    LISTS = MESH.FEMLISTS
    
    ###########################################################################
    if space == 'P1':
        
        LISTS['P1'] = {}
        LISTS['P1']['TRIG'] = {}
        
        LISTS['P1']['TRIG']['sizeM'] = MESH.np
        LISTS['P1']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        LISTS['P1']['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
        LISTS['P1']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
        
        LISTS['P1']['TRIG']['phi'] = {}
        LISTS['P1']['TRIG']['phi'][0] = lambda x,y: 1-x-y
        LISTS['P1']['TRIG']['phi'][1] = lambda x,y: x
        LISTS['P1']['TRIG']['phi'][2] = lambda x,y: y
        
        LISTS['P1']['TRIG']['dphi'] = {}
        LISTS['P1']['TRIG']['dphi'][0] = lambda x,y: np.r_[-1,-1]
        LISTS['P1']['TRIG']['dphi'][1] = lambda x,y: np.r_[ 1, 0]
        LISTS['P1']['TRIG']['dphi'][2] = lambda x,y: np.r_[ 0, 1]
        
        LISTS['P1']['B'] = {}
        LISTS['P1']['B']['phi'] = {}
        LISTS['P1']['B']['phi'][0] = lambda x: 1-x
        LISTS['P1']['B']['phi'][1] = lambda x: x
        LISTS['P1']['B']['qp_we_B'] = quadrature.one_d(order = 2)
        
        LISTS['P1']['TRIG']['LIST_DOF'] = MESH.t[:,0:3]
        LISTS['P1']['B']['LIST_DOF'] = MESH.e
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P2':
        
        LISTS['P2'] = {}
        LISTS['P2']['TRIG'] = {}   
        
        LISTS['P2']['TRIG']['sizeM'] = MESH.np + MESH.NoEdges
        LISTS['P2']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 4)
        LISTS['P2']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 2)
        
        LISTS['P2']['TRIG']['dphi'][0] = lambda x,y: np.r_[4*x+4*y-3, 4*x+4*y-3]
        LISTS['P2']['TRIG']['dphi'][1] = lambda x,y: np.r_[4*x-1, 0*x]
        LISTS['P2']['TRIG']['dphi'][2] = lambda x,y: np.r_[0*x, 4*y-1]
        LISTS['P2']['TRIG']['dphi'][3] = lambda x,y: np.r_[4*y, 4*x]
        LISTS['P2']['TRIG']['dphi'][4] = lambda x,y: np.r_[-4*y, -4*(x+2*y-1)]
        LISTS['P2']['TRIG']['dphi'][5] = lambda x,y: np.r_[-4*(2*x+y-1), -4*x]
        
        LISTS['P2']['B']['phi'][0] = lambda x: (1-x)*(1-2*x)
        LISTS['P2']['B']['phi'][1] = lambda x: x*(2*x-1)
        LISTS['P2']['B']['phi'][2] = lambda x: 4*x*(1-x)
        LISTS['P2']['B']['qp_we_B'] = quadrature.one_d(order = 5) # 4 would suffice
        
        LISTS['P2']['TRIG']['LIST_DOF'] = np.c_[MESH.t[:,0:3], MESH.np + MESH.TriangleToEdges]
        LISTS['P2']['B']['LIST_DOF'] = np.c_[MESH.e, MESH.np + MESH.Boundary_Edges]
        
    ###########################################################################