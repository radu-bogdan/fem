
from .. import quadrature
import numpy as np

def spaceInfo(MESH,space):
    
    LISTS = MESH.FEMLISTS
    
    ###########################################################################
    if space == 'P0': 
        
        LISTS['P0'] = {}
        LISTS['P0']['TET'] = {}
        
        LISTS['P0']['TET']['sizeM'] = MESH.nt
        LISTS['P0']['TET']['qp_we_M'] = quadrature.keast(order = 0)
        
        LISTS['P0']['TET']['phi'] = {}
        LISTS['P0']['TET']['phi'][0] = lambda x,y,z: 1+0*x+0*y+0*z
        
        LISTS['P0']['TET']['dphi'] = {}
        LISTS['P0']['TET']['dphi'][0] = lambda x,y,z: np.r_[ 0, 0, 0];
        
        LISTS['P0']['TET']['LIST_DOF'] = np.r_[:MESH.nt][:,None]
        
        # LISTS['P0']['B'] = {}
        # LISTS['P0']['B']['phi'] = {}
        # LISTS['P0']['B']['phi'][0] = lambda x: 1
        
        # LISTS['P0']['B']['LIST_DOF'] = np.r_[0:MESH.NoEdges]
        # LISTS['P0']['B']['sizeM'] = MESH.NoEdges
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P1':
        
        LISTS['P1'] = {}
        LISTS['P1']['TET'] = {}
        
        LISTS['P1']['TET']['sizeM'] = 4*MESH.nt
        LISTS['P1']['TET']['qp_we_M'] = quadrature.keast(order = 2)
        
        LISTS['P1']['TET']['phi'] = {}
        LISTS['P1']['TET']['phi'][0] = lambda x,y,z: 1-x-y-z
        LISTS['P1']['TET']['phi'][1] = lambda x,y,z: x
        LISTS['P1']['TET']['phi'][2] = lambda x,y,z: y
        LISTS['P1']['TET']['phi'][3] = lambda x,y,z: z
        
        LISTS['P1']['TET']['dphi'] = {}
        LISTS['P1']['TET']['dphi'][0] = lambda x,y,z: np.r_[-1,-1,-1]
        LISTS['P1']['TET']['dphi'][1] = lambda x,y,z: np.r_[ 1, 0, 0]
        LISTS['P1']['TET']['dphi'][2] = lambda x,y,z: np.r_[ 0, 1, 0]
        LISTS['P1']['TET']['dphi'][3] = lambda x,y,z: np.r_[ 0, 0, 1]
        
        LISTS['P1']['TET']['LIST_DOF'] = np.r_[:4*MESH.nt].reshape(MESH.nt,4)
        
        # LISTS['P1']['B'] = {}
        # LISTS['P1']['B']['phi'] = {}
        # LISTS['P1']['B']['phi'][0] = lambda x: 1-x
        # LISTS['P1']['B']['phi'][1] = lambda x: x
        
        # LISTS['P1']['B']['LIST_DOF'] = np.r_[0:2*MESH.NoEdges].reshape(MESH.NoEdges,2)
        # LISTS['P1']['B']['sizeM'] = 2*MESH.NoEdges
        
        # LISTS['P1']['B']['qp_we_B'] = quadrature.one_d(order = 2)
    ###########################################################################
    
    
    ###########################################################################
    # if space == 'P2':
        
    #     LISTS['P2'] = {}
    #     LISTS['P2']['TRIG'] = {}   
        
    #     LISTS['P2']['TRIG']['sizeM'] = 6*MESH.nt
    #     LISTS['P2']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 4)
    #     LISTS['P2']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 2)
        
        
    #     LISTS['P2']['TRIG']['phi'] = {}        
    #     LISTS['P2']['TRIG']['phi'][0] = lambda x,y: (1-x-y)*(1-2*x-2*y)
    #     LISTS['P2']['TRIG']['phi'][1] = lambda x,y: x*(2*x-1)
    #     LISTS['P2']['TRIG']['phi'][2] = lambda x,y: y*(2*y-1)
    #     LISTS['P2']['TRIG']['phi'][3] = lambda x,y: 4*x*y
    #     LISTS['P2']['TRIG']['phi'][4] = lambda x,y: 4*y*(1-x-y)
    #     LISTS['P2']['TRIG']['phi'][5] = lambda x,y: 4*x*(1-x-y)
        
    #     LISTS['P2']['TRIG']['dphi'] = {}
    #     LISTS['P2']['TRIG']['dphi'][0] = lambda x,y: np.r_[4*x+4*y-3, 4*x+4*y-3]
    #     LISTS['P2']['TRIG']['dphi'][1] = lambda x,y: np.r_[4*x-1, 0*x]
    #     LISTS['P2']['TRIG']['dphi'][2] = lambda x,y: np.r_[0*x, 4*y-1]
    #     LISTS['P2']['TRIG']['dphi'][3] = lambda x,y: np.r_[4*y, 4*x]
    #     LISTS['P2']['TRIG']['dphi'][4] = lambda x,y: np.r_[-4*y, -4*(x+2*y-1)]
    #     LISTS['P2']['TRIG']['dphi'][5] = lambda x,y: np.r_[-4*(2*x+y-1), -4*x]
        
    #     # LISTS['P2']['B'] = {}
    #     # LISTS['P2']['B']['phi'] = {}
    #     # LISTS['P2']['B']['phi'][0] = lambda x: (1-x)*(1-2*x)
    #     # LISTS['P2']['B']['phi'][1] = lambda x: x*(2*x-1)
    #     # LISTS['P2']['B']['phi'][2] = lambda x: 4*x*(1-x)
        
    #     # LISTS['P2']['B']['qp_we_B'] = quadrature.one_d(order = 5) # 4 would suffice
        
    #     LISTS['P2']['TRIG']['LIST_DOF'] =np.r_[0:6*MESH.nt].reshape(MESH.nt,6)
        
    ###########################################################################