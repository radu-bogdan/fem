
from .. import quadrature
import numpy as np

def spaceInfo(MESH,space):
    
    INFO = {}
    INFO['TRIG'] = {}
    INFO['TRIG']['phi'] = {}
    
    ###########################################################################
    if space == 'SP1':        
        INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
        INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
        INFO['sizeM'] = 2*MESH.NoEdges
        # INFO['sizeD'] = MESH.nt
        # INFO['qp_we_B'] = quadrature.one_d(order = 2)
        
        INFO['TRIG']['phi'][0] = lambda x,y: np.array([[0,x],[x,0]])
        INFO['TRIG']['phi'][1] = lambda x,y: np.array([[0,y],[y,0]])
        INFO['TRIG']['phi'][2] = lambda x,y: np.array([[2*y,-y],[-y,0]])
        INFO['TRIG']['phi'][3] = lambda x,y: np.array([[2*(1-x-y),-(1-x-y)],[-(1-x-y),0]])
        INFO['TRIG']['phi'][4] = lambda x,y: np.array([[0,-(1-x-y)],[-(1-x-y),2*(1-x-y)]])
        INFO['TRIG']['phi'][5] = lambda x,y: np.array([[0,-x],[-x,2*x]])
    ###########################################################################
    
    return INFO