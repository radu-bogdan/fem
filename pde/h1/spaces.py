
from .. import quadrature

def spaceInfo(MESH,space):
    
    INFO = {}
    INFO['TRIG'] = {}
    
    ###########################################################################
    if space == 'P1': 
        INFO['TRIG']['space'] = 'P1'
        # INFO['TRIG']['space_dx'] = 'P0'
        INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        INFO['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
        INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
        INFO['sizeM'] = MESH.np
        INFO['sizeD'] = MESH.nt
        INFO['qp_we_B'] = quadrature.one_d(order = 2)
    ###########################################################################
    
    
    ###########################################################################
    if space == 'P2': 
        INFO['TRIG']['space'] = 'P2'
        INFO['TRIG']['space_dx'] = 'P1'
        INFO['TRIG']['qp_we_M'] = quadrature.dunavant(order = 4)        
        INFO['TRIG']['qp_we_K'] = quadrature.dunavant(order = 2)
        INFO['sizeM'] = MESH.np + MESH.NoEdges
        INFO['sizeD'] = 3*MESH.nt
        INFO['qp_we_B'] = quadrature.one_d(order = 5) # 4 would suffice
    ###########################################################################
    
    return INFO