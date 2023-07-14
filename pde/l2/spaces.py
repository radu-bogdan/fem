
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
        
        LISTS['P1']['TRIG']['LIST_DOF'] = np.r_[0:3*MESH.nt].reshape(MESH.nt,3)
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
    