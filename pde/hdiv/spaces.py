
from .. import quadrature
import numpy as np

def spaceInfo(MESH,space):
    
    LISTS = MESH.FEMLISTS
    
    ###########################################################################
    if space == 'RT0':
        
        LISTS['RT0'] = {}
        LISTS['RT0']['TRIG'] = {}
        
        LISTS['RT0']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        LISTS['RT0']['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
        LISTS['RT0']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
        
        LISTS['RT0']['TRIG']['sizeM'] = MESH.NoEdges
        
        LISTS['RT0']['TRIG']['phi'] = {}
        LISTS['RT0']['TRIG']['phi'][0] = lambda x,y: np.r_[x,y]
        LISTS['RT0']['TRIG']['phi'][1] = lambda x,y: np.r_[x-1,y]
        LISTS['RT0']['TRIG']['phi'][2] = lambda x,y: np.r_[x,y-1]

        LISTS['RT0']['TRIG']['divphi'] = {}
        LISTS['RT0']['TRIG']['divphi'][0] = lambda x,y: 2+0*x
        LISTS['RT0']['TRIG']['divphi'][1] = lambda x,y: 2+0*x
        LISTS['RT0']['TRIG']['divphi'][2] = lambda x,y: 2+0*x
        
        LISTS['RT0']['TRIG']['LIST_DOF'] = MESH.TriangleToEdges
        LISTS['RT0']['TRIG']['DIRECTION_DOF'] = MESH.EdgeDirectionTrig

        LISTS['RT0']['B'] = {}
        LISTS['RT0']['B']['phi'] = {}
        LISTS['RT0']['B']['phi'][0] = lambda x: 1
        LISTS['RT0']['B']['qp_we_B'] = quadrature.one_d(order = 0)
        
        LISTS['RT0']['B']['LIST_DOF'] = MESH.Boundary_Edges
    ###########################################################################

    
    ###########################################################################
    if space == 'EJ1':
        
        LISTS['EJ1'] = {}
        LISTS['EJ1']['TRIG'] = {}
        
        LISTS['EJ1']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 4)
        LISTS['EJ1']['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = '2m')
        LISTS['EJ1']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 2)
        
        LISTS['EJ1']['sizeM'] = MESH.NoEdges + 3*MESH.nt
        
        LISTS['EJ1']['TRIG']['phi'] = {}
        LISTS['EJ1']['TRIG']['phi'][0] = lambda x,y: np.r_[x,y] # RT0 basis
        LISTS['EJ1']['TRIG']['phi'][1] = lambda x,y: np.r_[x-1,y] # RT0 basis
        LISTS['EJ1']['TRIG']['phi'][2] = lambda x,y: np.r_[x,y-1] # RT0 basis
        
        LISTS['EJ1']['TRIG']['divphi'][0] = lambda x,y: 2+0*x # RT0 basis
        LISTS['EJ1']['TRIG']['divphi'][1] = lambda x,y: 2+0*x # RT0 basis
        LISTS['EJ1']['TRIG']['divphi'][2] = lambda x,y: 2+0*x # RT0 basis
        
        LISTS['EJ1']['TRIG']['phi'][3] = lambda x,y: np.r_[4*x*y,-4*x*y]
        LISTS['EJ1']['TRIG']['phi'][4] = lambda x,y: np.r_[0*x,4*y*(1-x-y)]
        LISTS['EJ1']['TRIG']['phi'][5] = lambda x,y: np.r_[4*x*(1-x-y),0*y]
        
        LISTS['EJ1']['TRIG']['divphi'][3] = lambda x,y: 4*y-4*x
        LISTS['EJ1']['TRIG']['divphi'][4] = lambda x,y: 4-4*x-8*y
        LISTS['EJ1']['TRIG']['divphi'][5] = lambda x,y: 4-8*x-4*y
        
        LISTS['EJ1']['TRIG']['phi'][0] = lambda x,y: LISTS['EJ1']['TRIG']['phi'][0](x,y) -1/2*LISTS['EJ1']['TRIG']['phi'][4](x,y) -1/2*LISTS['EJ1']['TRIG']['phi'][5](x,y)
        LISTS['EJ1']['TRIG']['phi'][1] = lambda x,y: LISTS['EJ1']['TRIG']['phi'][1](x,y) +1/2*LISTS['EJ1']['TRIG']['phi'][5](x,y) +1/2*LISTS['EJ1']['TRIG']['phi'][3](x,y)
        LISTS['EJ1']['TRIG']['phi'][2] = lambda x,y: LISTS['EJ1']['TRIG']['phi'][2](x,y) -1/2*LISTS['EJ1']['TRIG']['phi'][3](x,y) +1/2*LISTS['EJ1']['TRIG']['phi'][4](x,y)
        
        LISTS['EJ1']['TRIG']['divphi'][0] = lambda x,y: LISTS['EJ1']['TRIG']['divphi'][0](x,y) -1/2*LISTS['EJ1']['TRIG']['divphi'][4](x,y) -1/2*LISTS['EJ1']['TRIG']['divphi'][5](x,y)
        LISTS['EJ1']['TRIG']['divphi'][1] = lambda x,y: LISTS['EJ1']['TRIG']['divphi'][1](x,y) +1/2*LISTS['EJ1']['TRIG']['divphi'][5](x,y) +1/2*LISTS['EJ1']['TRIG']['divphi'][3](x,y)
        LISTS['EJ1']['TRIG']['divphi'][2] = lambda x,y: LISTS['EJ1']['TRIG']['divphi'][2](x,y) -1/2*LISTS['EJ1']['TRIG']['divphi'][3](x,y) +1/2*LISTS['EJ1']['TRIG']['divphi'][4](x,y)       
        
        
        LISTS['EJ1']['TRIG']['LIST_DOF'] = np.c_[MESH.TriangleToEdges,
                                                 range(MESH.NoEdges+0,MESH.NoEdges+3*MESH.nt,3),
                                                 range(MESH.NoEdges+1,MESH.NoEdges+3*MESH.nt,3),
                                                 range(MESH.NoEdges+2,MESH.NoEdges+3*MESH.nt,3)]
        
        LISTS['EJ1']['TRIG']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTrig,
                                                      np.ones(MESH.nt),
                                                      np.ones(MESH.nt),
                                                      np.ones(MESH.nt)]

        LISTS['EJ1']['B'] = {}
        LISTS['EJ1']['B']['phi'] = {}
        LISTS['EJ1']['B']['phi'][0] = lambda x: 1
        LISTS['EJ1']['B']['qp_we_B'] = quadrature.one_d(order = 0)
        
        LISTS['EJ1']['B']['LIST_DOF'] = MESH.Boundary_Edges
    ###########################################################################
    
    
    ###########################################################################
    if space == 'BDM1':
        
        LISTS['BDM1'] = {}
        LISTS['BDM1']['TRIG'] = {}
        
        LISTS['BDM1']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        LISTS['BDM1']['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
        LISTS['BDM1']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
        
        LISTS['BDM1']['TRIG']['sizeM'] = 2*MESH.NoEdges
        
        LISTS['BDM1']['TRIG']['phi'] = {}
        LISTS['BDM1']['TRIG']['phi'][0] = lambda x,y: np.r_[x,0*y]
        LISTS['BDM1']['TRIG']['phi'][1] = lambda x,y: np.r_[0*x,y]
        LISTS['BDM1']['TRIG']['phi'][2] = lambda x,y: np.r_[-y,y]
        LISTS['BDM1']['TRIG']['phi'][3] = lambda x,y: np.r_[x+y-1,0*y]
        LISTS['BDM1']['TRIG']['phi'][4] = lambda x,y: np.r_[0*x,x+y-1]
        LISTS['BDM1']['TRIG']['phi'][5] = lambda x,y: np.r_[x,-x]
        
        LISTS['BDM1']['TRIG']['divphi'] = {}
        LISTS['BDM1']['TRIG']['divphi'][0] = lambda x,y: 1+0*x
        LISTS['BDM1']['TRIG']['divphi'][1] = lambda x,y: 1+0*x
        LISTS['BDM1']['TRIG']['divphi'][2] = lambda x,y: 1+0*x
        LISTS['BDM1']['TRIG']['divphi'][3] = lambda x,y: 1+0*x
        LISTS['BDM1']['TRIG']['divphi'][4] = lambda x,y: 1+0*x
        LISTS['BDM1']['TRIG']['divphi'][5] = lambda x,y: 1+0*x
        
        LISTS['BDM1']['TRIG']['LIST_DOF'] = np.c_[2*MESH.TriangleToEdges[:,0]   -1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                  2*MESH.TriangleToEdges[:,0]+1 +1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                  2*MESH.TriangleToEdges[:,1]   -1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                  2*MESH.TriangleToEdges[:,1]+1 +1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                  2*MESH.TriangleToEdges[:,2]   -1/2*(MESH.EdgeDirectionTrig[:,2]-1),
                                                  2*MESH.TriangleToEdges[:,2]+1 +1/2*(MESH.EdgeDirectionTrig[:,2]-1)]
    
        LISTS['BDM1']['TRIG']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTrig[:,0],
                                                       MESH.EdgeDirectionTrig[:,0],
                                                       MESH.EdgeDirectionTrig[:,1],
                                                       MESH.EdgeDirectionTrig[:,1],
                                                       MESH.EdgeDirectionTrig[:,2],
                                                       MESH.EdgeDirectionTrig[:,2]]
        
        LISTS['BDM1']['B'] = {}
        LISTS['BDM1']['B']['phi'] = {}
        LISTS['BDM1']['B']['phi'][0] = lambda x: 1-x
        LISTS['BDM1']['B']['phi'][1] = lambda x: x
        LISTS['BDM1']['B']['qp_we_B'] = quadrature.one_d(order = 2)
        
        LISTS['BDM1']['B']['LIST_DOF'] = np.c_[2*MESH.Boundary_Edges,
                                               2*MESH.Boundary_Edges + 1]
    ###########################################################################
    
    
    
    ###########################################################################
    if space == 'RT1':
        
        LISTS['RT1'] = {}
        LISTS['RT1']['TRIG'] = {}
        
        LISTS['RT1']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 4)
        LISTS['RT1']['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = '2l')
        LISTS['RT1']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 2)
        
        LISTS['RT1']['TRIG']['sizeM'] = 2*MESH.NoEdges + 2*MESH.nt
        
        LISTS['RT1']['TRIG']['phi'] = {}
        LISTS['RT1']['TRIG']['phi'][6] = lambda x,y: np.r_[x*y,y*(y-1)]
        LISTS['RT1']['TRIG']['phi'][7] = lambda x,y: np.r_[x*(x-1),x*y]

        LISTS['RT1']['TRIG']['divphi'] = {}
        LISTS['RT1']['TRIG']['divphi'][6] = lambda x,y: 3*y-1
        LISTS['RT1']['TRIG']['divphi'][7] = lambda x,y: 3*x-1

        LISTS['RT1']['TRIG']['phi'][0] = lambda x,y: LISTS['RT1']['TRIG']['phi'][0](x,y) +1*LISTS['RT1']['TRIG']['phi'][6](x,y) +2*LISTS['RT1']['TRIG']['phi'][7](x,y)
        LISTS['RT1']['TRIG']['phi'][1] = lambda x,y: LISTS['RT1']['TRIG']['phi'][1](x,y) +2*LISTS['RT1']['TRIG']['phi'][6](x,y) +1*LISTS['RT1']['TRIG']['phi'][7](x,y)
        LISTS['RT1']['TRIG']['phi'][2] = lambda x,y: LISTS['RT1']['TRIG']['phi'][2](x,y) +1*LISTS['RT1']['TRIG']['phi'][6](x,y) -1*LISTS['RT1']['TRIG']['phi'][7](x,y)
        LISTS['RT1']['TRIG']['phi'][3] = lambda x,y: LISTS['RT1']['TRIG']['phi'][3](x,y) -1*LISTS['RT1']['TRIG']['phi'][6](x,y) -2*LISTS['RT1']['TRIG']['phi'][7](x,y)
        LISTS['RT1']['TRIG']['phi'][4] = lambda x,y: LISTS['RT1']['TRIG']['phi'][4](x,y) -2*LISTS['RT1']['TRIG']['phi'][6](x,y) -1*LISTS['RT1']['TRIG']['phi'][7](x,y)
        LISTS['RT1']['TRIG']['phi'][5] = lambda x,y: LISTS['RT1']['TRIG']['phi'][5](x,y) -1*LISTS['RT1']['TRIG']['phi'][6](x,y) +1*LISTS['RT1']['TRIG']['phi'][7](x,y)

        LISTS['RT1']['TRIG']['divphi'][0] = lambda x,y: LISTS['RT1']['TRIG']['divphi'][0](x,y) +1*LISTS['RT1']['TRIG']['divphi'][6](x,y) +2*LISTS['RT1']['TRIG']['divphi'][7](x,y)
        LISTS['RT1']['TRIG']['divphi'][1] = lambda x,y: LISTS['RT1']['TRIG']['divphi'][1](x,y) +2*LISTS['RT1']['TRIG']['divphi'][6](x,y) +1*LISTS['RT1']['TRIG']['divphi'][7](x,y)
        LISTS['RT1']['TRIG']['divphi'][2] = lambda x,y: LISTS['RT1']['TRIG']['divphi'][2](x,y) +1*LISTS['RT1']['TRIG']['divphi'][6](x,y) -1*LISTS['RT1']['TRIG']['divphi'][7](x,y)
        LISTS['RT1']['TRIG']['divphi'][3] = lambda x,y: LISTS['RT1']['TRIG']['divphi'][3](x,y) -1*LISTS['RT1']['TRIG']['divphi'][6](x,y) -2*LISTS['RT1']['TRIG']['divphi'][7](x,y)
        LISTS['RT1']['TRIG']['divphi'][4] = lambda x,y: LISTS['RT1']['TRIG']['divphi'][4](x,y) -2*LISTS['RT1']['TRIG']['divphi'][6](x,y) -1*LISTS['RT1']['TRIG']['divphi'][7](x,y)
        LISTS['RT1']['TRIG']['divphi'][5] = lambda x,y: LISTS['RT1']['TRIG']['divphi'][5](x,y) -1*LISTS['RT1']['TRIG']['divphi'][6](x,y) +1*LISTS['RT1']['TRIG']['divphi'][7](x,y)        
        
        LISTS['RT1']['TRIG']['LIST_DOF'] = np.c_[2*MESH.TriangleToEdges[:,0]   -1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                 2*MESH.TriangleToEdges[:,0]+1 +1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                 2*MESH.TriangleToEdges[:,1]   -1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                 2*MESH.TriangleToEdges[:,1]+1 +1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                 2*MESH.TriangleToEdges[:,2]   -1/2*(MESH.EdgeDirectionTrig[:,2]-1),
                                                 2*MESH.TriangleToEdges[:,2]+1 +1/2*(MESH.EdgeDirectionTrig[:,2]-1),
                                                 range(2*MESH.NoEdges+0,2*MESH.NoEdges+2*MESH.nt,2),
                                                 range(2*MESH.NoEdges+1,2*MESH.NoEdges+2*MESH.nt,2)]
    
        LISTS['RT1']['TRIG']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTrig[:,0],
                                                      MESH.EdgeDirectionTrig[:,0],
                                                      MESH.EdgeDirectionTrig[:,1],
                                                      MESH.EdgeDirectionTrig[:,1],
                                                      MESH.EdgeDirectionTrig[:,2],
                                                      MESH.EdgeDirectionTrig[:,2],
                                                      np.ones(MESH.nt),
                                                      np.ones(MESH.nt)]
    ###########################################################################