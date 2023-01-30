
from .. import quadrature

def spaceInfo(MESH,space):
    
    INFO = {}
    INFO['TRIG'] = {}
    
    ###########################################################################
    if space == 'P0': 
        INFO['TRIG']['space'] = 'P0'
        INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 0)
        INFO['sizeM'] = MESH.nt
        INFO['qp_we_B'] = quadrature.one_d(order = 2)
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P1d': 
        INFO['TRIG']['space'] = 'P1d'
        INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        INFO['sizeM'] = 3*MESH.nt
        INFO['qp_we_B'] = quadrature.one_d(order = 2)
    ###########################################################################
    
    return INFO