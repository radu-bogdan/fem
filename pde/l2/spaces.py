
from .. import quadrature
import numpy as np

def spaceInfo(MESH,space):
    
    LISTS = MESH.FEMLISTS
    
    ###########################################################################
    if space == 'P0': 
        
        LISTS['P0'] = {}
        LISTS['P0']['TRIG'] = {}
        
        LISTS['P0']['TRIG']['sizeM'] = MESH.nt
        LISTS['P0']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 0)
        
        LISTS['P0']['B'] = {}
        LISTS['P0']['B']['phi'] = {}
        LISTS['P0']['B']['phi'][0] = lambda x: 1
        
        LISTS['P0']['B']['LIST_DOF'] = np.r_[0:MESH.NoEdges]
        LISTS['P0']['B']['sizeM'] = MESH.NoEdges
        
        LISTS['P0']['TRIG']['phi'] = {}
        LISTS['P0']['TRIG']['phi'][0] = lambda x,y: 1+0*x+0*y
        
        LISTS['P0']['TRIG']['dphi'] = {}
        LISTS['P0']['TRIG']['dphi'][0] = lambda x,y: np.r_[ 0, 0];
        
        LISTS['P0']['TRIG']['LIST_DOF'] = np.r_[0:MESH.nt][:,None]
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P1':
        
        LISTS['P1'] = {}
        LISTS['P1']['TRIG'] = {}
        
        LISTS['P1']['TRIG']['sizeM'] = 3*MESH.nt
        LISTS['P1']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        
        LISTS['P1']['B'] = {}
        LISTS['P1']['B']['phi'] = {}
        LISTS['P1']['B']['phi'][0] = lambda x: 1-x
        LISTS['P1']['B']['phi'][1] = lambda x: x
        
        LISTS['P1']['B']['LIST_DOF'] = np.r_[0:2*MESH.NoEdges].reshape(MESH.NoEdges,2)
        LISTS['P1']['B']['sizeM'] = 2*MESH.NoEdges
        
        LISTS['P1']['B']['qp_we_B'] = quadrature.one_d(order = 2)
        
        LISTS['P1']['TRIG']['phi'] = {}
        LISTS['P1']['TRIG']['phi'][0] = lambda x,y: 1-x-y
        LISTS['P1']['TRIG']['phi'][1] = lambda x,y: x
        LISTS['P1']['TRIG']['phi'][2] = lambda x,y: y
        
        LISTS['P1']['TRIG']['dphi'] = {}
        LISTS['P1']['TRIG']['dphi'][0] = lambda x,y: np.r_[-1,-1]
        LISTS['P1']['TRIG']['dphi'][1] = lambda x,y: np.r_[ 1, 0]
        LISTS['P1']['TRIG']['dphi'][2] = lambda x,y: np.r_[ 0, 1]
        
        LISTS['P1']['TRIG']['LIST_DOF'] = np.r_[0:3*MESH.nt].reshape(MESH.nt,3)
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P2':
        
        LISTS['P2'] = {}
        LISTS['P2']['TRIG'] = {}   
        
        LISTS['P2']['TRIG']['sizeM'] = 6*MESH.nt
        LISTS['P2']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 4)
        LISTS['P2']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 2)
        
        
        LISTS['P2']['TRIG']['phi'] = {}        
        LISTS['P2']['TRIG']['phi'][0] = lambda x,y: (1-x-y)*(1-2*x-2*y)
        LISTS['P2']['TRIG']['phi'][1] = lambda x,y: x*(2*x-1)
        LISTS['P2']['TRIG']['phi'][2] = lambda x,y: y*(2*y-1)
        LISTS['P2']['TRIG']['phi'][3] = lambda x,y: 4*x*y
        LISTS['P2']['TRIG']['phi'][4] = lambda x,y: 4*y*(1-x-y)
        LISTS['P2']['TRIG']['phi'][5] = lambda x,y: 4*x*(1-x-y)
        
        LISTS['P2']['TRIG']['dphi'] = {}
        LISTS['P2']['TRIG']['dphi'][0] = lambda x,y: np.r_[4*x+4*y-3, 4*x+4*y-3]
        LISTS['P2']['TRIG']['dphi'][1] = lambda x,y: np.r_[4*x-1, 0*x]
        LISTS['P2']['TRIG']['dphi'][2] = lambda x,y: np.r_[0*x, 4*y-1]
        LISTS['P2']['TRIG']['dphi'][3] = lambda x,y: np.r_[4*y, 4*x]
        LISTS['P2']['TRIG']['dphi'][4] = lambda x,y: np.r_[-4*y, -4*(x+2*y-1)]
        LISTS['P2']['TRIG']['dphi'][5] = lambda x,y: np.r_[-4*(2*x+y-1), -4*x]
        
        # LISTS['P2']['B'] = {}
        # LISTS['P2']['B']['phi'] = {}
        # LISTS['P2']['B']['phi'][0] = lambda x: (1-x)*(1-2*x)
        # LISTS['P2']['B']['phi'][1] = lambda x: x*(2*x-1)
        # LISTS['P2']['B']['phi'][2] = lambda x: 4*x*(1-x)
        
        # LISTS['P2']['B']['qp_we_B'] = quadrature.one_d(order = 5) # 4 would suffice
        
        LISTS['P2']['TRIG']['LIST_DOF'] =np.r_[0:6*MESH.nt].reshape(MESH.nt,6)
        
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P1_orth_divN1':
        
        LISTS[space] = {}
        LISTS[space]['TRIG'] = {}
        
        LISTS[space]['TRIG']['sizeM'] = 3*MESH.nt
        LISTS[space]['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        
        LISTS[space]['B'] = {}
        LISTS[space]['B']['phi'] = {}
        LISTS[space]['B']['phi'][0] = lambda x: 1-x
        LISTS[space]['B']['phi'][1] = lambda x: x
        
        LISTS[space]['B']['LIST_DOF'] = np.r_[0:2*MESH.NoEdges].reshape(MESH.NoEdges,2)
        LISTS[space]['B']['sizeM'] = 2*MESH.NoEdges
        
        LISTS[space]['B']['qp_we_B'] = quadrature.one_d(order = 2)
        
        LISTS[space]['TRIG']['phi'] = {}
        LISTS[space]['TRIG']['phi'][0] = lambda x,y: -12*x-12*y+11
        LISTS[space]['TRIG']['phi'][1] = lambda x,y:  12*x-1
        LISTS[space]['TRIG']['phi'][2] = lambda x,y:  12*y-1
        
        # LISTS[space]['TRIG']['phi'][0] = lambda x,y: -12*x+7
        # LISTS[space]['TRIG']['phi'][1] = lambda x,y: -12*y+7
        # LISTS[space]['TRIG']['phi'][2] = lambda x,y:  12*x+12*y-5
        
        LISTS[space]['TRIG']['LIST_DOF'] = np.r_[0:3*MESH.nt].reshape(MESH.nt,3)
    ###########################################################################
    