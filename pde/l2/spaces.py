
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
        
        LISTS['P0']['B']['qp_we_B'] = quadrature.one_d(order = 0)
        
        LISTS['P0']['TRIG']['phi'] = {}
        LISTS['P0']['TRIG']['phi'][0] = lambda x,y: 1+0*x+0*y
        
        LISTS['P0']['TRIG']['LIST_DOF'] = np.r_[0:MESH.nt][:,None]
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P1d':
        
        LISTS['P1d'] = {}
        LISTS['P1d']['TRIG'] = {}
        
        LISTS['P1d']['TRIG']['sizeM'] = 3*MESH.nt
        LISTS['P1d']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        
        LISTS['P1d']['B'] = {}
        LISTS['P1d']['B']['qp_we_B'] = quadrature.one_d(order = 2)
        
        LISTS['P1d']['TRIG']['phi'][0] = lambda x,y: 1-x-y
        LISTS['P1d']['TRIG']['phi'][1] = lambda x,y: x
        LISTS['P1d']['TRIG']['phi'][2] = lambda x,y: y
        
        LISTS['P1d']['TRIG']['LIST_DOF'] = np.r_[0:3*MESH.nt].reshape(MESH.nt,3)
    ###########################################################################
    