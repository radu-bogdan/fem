
from .. import quadrature
import numpy as np

def spaceInfo(MESH,space):
    
    LISTS = MESH.FEMLISTS
    
    ###########################################################################
    if space == 'N0':
        
        LISTS['N0'] = {}
        LISTS['N0']['TRIG'] = {}
        
        LISTS['N0']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        LISTS['N0']['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
        LISTS['N0']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
        
        LISTS['N0']['TRIG']['sizeM'] = MESH.NoEdges
        
        LISTS['N0']['TRIG']['phi'] = {}
        LISTS['N0']['TRIG']['phi'][0] = lambda x,y: np.r_[-y,x]
        LISTS['N0']['TRIG']['phi'][1] = lambda x,y: np.r_[-y,x-1]
        LISTS['N0']['TRIG']['phi'][2] = lambda x,y: np.r_[-y+1,x]
        
        LISTS['N0']['TRIG']['curlphi'] = {}
        LISTS['N0']['TRIG']['curlphi'][0] = lambda x,y: 2+0*x
        LISTS['N0']['TRIG']['curlphi'][1] = lambda x,y: 2+0*x
        LISTS['N0']['TRIG']['curlphi'][2] = lambda x,y: 2+0*x
        
        LISTS['N0']['TRIG']['LIST_DOF'] = MESH.TriangleToEdges
        LISTS['N0']['TRIG']['DIRECTION_DOF'] = MESH.EdgeDirectionTrig

        LISTS['N0']['B'] = {}
        LISTS['N0']['B']['phi'] = {}
        LISTS['N0']['B']['phi'][0] = lambda x: 1
        LISTS['N0']['B']['qp_we_B'] = quadrature.one_d(order = 0)
        
        LISTS['N0']['B']['LIST_DOF'] = MESH.Boundary_Edges
        LISTS['N0']['B']['LIST_DOF_E1'] = MESH.IntEdgesToTriangles[:,0]
        LISTS['N0']['B']['LIST_DOF_E1'] = MESH.IntEdgesToTriangles[:,1]
        
        LISTS['N0']['TRIG']['phidual'] = {}
        LISTS['N0']['TRIG']['phidual'][0] = lambda x: 2+0*x
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
        LISTS['EJ1']['TRIG']['phi'][0] = lambda x,y: np.r_[-y,x] # RT0 basis
        LISTS['EJ1']['TRIG']['phi'][1] = lambda x,y: np.r_[-y,x-1] # RT0 basis
        LISTS['EJ1']['TRIG']['phi'][2] = lambda x,y: np.r_[-y+1,x] # RT0 basis
        
        LISTS['EJ1']['TRIG']['curlphi'] = {}   
        LISTS['EJ1']['TRIG']['curlphi'][0] = lambda x,y: 2+0*x # RT0 basis
        LISTS['EJ1']['TRIG']['curlphi'][1] = lambda x,y: 2+0*x # RT0 basis
        LISTS['EJ1']['TRIG']['curlphi'][2] = lambda x,y: 2+0*x # RT0 basis
        
        LISTS['EJ1']['TRIG']['phi'][3] = lambda x,y: np.r_[4*x*y,4*x*y]
        LISTS['EJ1']['TRIG']['phi'][4] = lambda x,y: np.r_[-4*y*(1-x-y),0*y]
        LISTS['EJ1']['TRIG']['phi'][5] = lambda x,y: np.r_[0*x,4*x*(1-x-y)]
        
        LISTS['EJ1']['TRIG']['curlphi'][3] = lambda x,y: 4*y-4*x
        LISTS['EJ1']['TRIG']['curlphi'][4] = lambda x,y: 4-4*x-8*y
        LISTS['EJ1']['TRIG']['curlphi'][5] = lambda x,y: 4-8*x-4*y
        
        LISTS['EJ1']['TRIG']['phi'][0] = lambda x,y: LISTS['EJ1']['TRIG']['phi'][0](x,y) -1/2*LISTS['EJ1']['TRIG']['phi'][4](x,y) -1/2*LISTS['EJ1']['TRIG']['phi'][5](x,y)
        LISTS['EJ1']['TRIG']['phi'][1] = lambda x,y: LISTS['EJ1']['TRIG']['phi'][1](x,y) +1/2*LISTS['EJ1']['TRIG']['phi'][5](x,y) +1/2*LISTS['EJ1']['TRIG']['phi'][3](x,y)
        LISTS['EJ1']['TRIG']['phi'][2] = lambda x,y: LISTS['EJ1']['TRIG']['phi'][2](x,y) -1/2*LISTS['EJ1']['TRIG']['phi'][3](x,y) +1/2*LISTS['EJ1']['TRIG']['phi'][4](x,y)
        
        LISTS['EJ1']['TRIG']['curlphi'][0] = lambda x,y: LISTS['EJ1']['TRIG']['curlphi'][0](x,y) -1/2*LISTS['EJ1']['TRIG']['curlphi'][4](x,y) -1/2*LISTS['EJ1']['TRIG']['curlphi'][5](x,y)
        LISTS['EJ1']['TRIG']['curlphi'][1] = lambda x,y: LISTS['EJ1']['TRIG']['curlphi'][1](x,y) +1/2*LISTS['EJ1']['TRIG']['curlphi'][5](x,y) +1/2*LISTS['EJ1']['TRIG']['curlphi'][3](x,y)
        LISTS['EJ1']['TRIG']['curlphi'][2] = lambda x,y: LISTS['EJ1']['TRIG']['curlphi'][2](x,y) -1/2*LISTS['EJ1']['TRIG']['curlphi'][3](x,y) +1/2*LISTS['EJ1']['TRIG']['curlphi'][4](x,y)       
        
        
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
    if space == 'NC1':
        
        LISTS['NC1'] = {}
        LISTS['NC1']['TRIG'] = {}
        
        LISTS['NC1']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        LISTS['NC1']['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
        LISTS['NC1']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
        
        LISTS['NC1']['TRIG']['sizeM'] = 2*MESH.NoEdges
        
        LISTS['NC1']['TRIG']['phi'] = {}
        LISTS['NC1']['TRIG']['phi'][0] = lambda x,y: np.r_[0*x,x]
        LISTS['NC1']['TRIG']['phi'][1] = lambda x,y: np.r_[-y,0*x]
        LISTS['NC1']['TRIG']['phi'][2] = lambda x,y: np.r_[-y,-y]
        LISTS['NC1']['TRIG']['phi'][3] = lambda x,y: np.r_[0*x,x+y-1]
        LISTS['NC1']['TRIG']['phi'][4] = lambda x,y: np.r_[-x-y+1,0*y]
        LISTS['NC1']['TRIG']['phi'][5] = lambda x,y: np.r_[x,x]
        
        LISTS['NC1']['TRIG']['curlphi'] = {}
        LISTS['NC1']['TRIG']['curlphi'][0] = lambda x,y: 1+0*x
        LISTS['NC1']['TRIG']['curlphi'][1] = lambda x,y: 1+0*x
        LISTS['NC1']['TRIG']['curlphi'][2] = lambda x,y: 1+0*x
        LISTS['NC1']['TRIG']['curlphi'][3] = lambda x,y: 1+0*x
        LISTS['NC1']['TRIG']['curlphi'][4] = lambda x,y: 1+0*x
        LISTS['NC1']['TRIG']['curlphi'][5] = lambda x,y: 1+0*x
        
        LISTS['NC1']['TRIG']['LIST_DOF'] = np.c_[2*MESH.TriangleToEdges[:,0]   -1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                 2*MESH.TriangleToEdges[:,0]+1 +1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                 2*MESH.TriangleToEdges[:,1]   -1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                 2*MESH.TriangleToEdges[:,1]+1 +1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                 2*MESH.TriangleToEdges[:,2]   -1/2*(MESH.EdgeDirectionTrig[:,2]-1),
                                                 2*MESH.TriangleToEdges[:,2]+1 +1/2*(MESH.EdgeDirectionTrig[:,2]-1)]
    
        LISTS['NC1']['TRIG']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTrig[:,0],
                                                      MESH.EdgeDirectionTrig[:,0],
                                                      MESH.EdgeDirectionTrig[:,1],
                                                      MESH.EdgeDirectionTrig[:,1],
                                                      MESH.EdgeDirectionTrig[:,2],
                                                      MESH.EdgeDirectionTrig[:,2]]
        
        LISTS['NC1']['B'] = {}
        LISTS['NC1']['B']['phi'] = {}
        LISTS['NC1']['B']['phi'][0] = lambda x: 1-x
        LISTS['NC1']['B']['phi'][1] = lambda x: x
        LISTS['NC1']['B']['qp_we_B'] = quadrature.one_d(order = 2)
        
        LISTS['NC1']['B']['LIST_DOF'] = np.c_[2*MESH.Boundary_Edges,
                                              2*MESH.Boundary_Edges + 1]
        
        LISTS['NC1']['B']['LIST_DOF_E1'] = np.c_[2*MESH.IntEdgesToTriangles[:,0],
                                                 2*MESH.IntEdgesToTriangles[:,0]+1]
        LISTS['NC1']['B']['LIST_DOF_E1'] = np.c_[2*MESH.IntEdgesToTriangles[:,1],
                                                 2*MESH.IntEdgesToTriangles[:,1]+1]

                                              
        LISTS['NC1']['B']['LIST_DOF_E'] = np.c_[2*MESH.NonSingle_Edges,
                                                2*MESH.NonSingle_Edges + 1]
        
        LISTS['NC1']['TRIG']['phidual'] = {}
        LISTS['NC1']['TRIG']['phidual'][0] = lambda x: 6*x-2 # dual to x and 1-x on the edge...
        LISTS['NC1']['TRIG']['phidual'][1] = lambda x: -6*x+4
    ###########################################################################
    
    
    
    ###########################################################################
    if space == 'N1':
        
        LISTS['N1'] = {}
        LISTS['N1']['TRIG'] = {}
        
        LISTS['N1']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 4)
        LISTS['N1']['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = '2l')
        LISTS['N1']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 2)
        
        LISTS['N1']['TRIG']['sizeM'] = 2*MESH.NoEdges + 2*MESH.nt
        
        LISTS['N1']['TRIG']['phi'] = {}
        LISTS['N1']['TRIG']['phi'][6] = lambda x,y: np.r_[-y*(y-1),x*y]
        LISTS['N1']['TRIG']['phi'][7] = lambda x,y: np.r_[-x*y,x*(x-1)]

        LISTS['N1']['TRIG']['curlphi'] = {}
        LISTS['N1']['TRIG']['curlphi'][6] = lambda x,y: 3*y-1
        LISTS['N1']['TRIG']['curlphi'][7] = lambda x,y: 3*x-1

        LISTS['N1']['TRIG']['phi'][0] = lambda x,y: np.r_[0*x,x] +1*LISTS['N1']['TRIG']['phi'][6](x,y) +2*LISTS['N1']['TRIG']['phi'][7](x,y)
        LISTS['N1']['TRIG']['phi'][1] = lambda x,y: np.r_[-y,0*x] +2*LISTS['N1']['TRIG']['phi'][6](x,y) +1*LISTS['N1']['TRIG']['phi'][7](x,y)
        LISTS['N1']['TRIG']['phi'][2] = lambda x,y: np.r_[-y,-y] +1*LISTS['N1']['TRIG']['phi'][6](x,y) -1*LISTS['N1']['TRIG']['phi'][7](x,y)
        LISTS['N1']['TRIG']['phi'][3] = lambda x,y: np.r_[0*x,x+y-1] -1*LISTS['N1']['TRIG']['phi'][6](x,y) -2*LISTS['N1']['TRIG']['phi'][7](x,y)
        LISTS['N1']['TRIG']['phi'][4] = lambda x,y: np.r_[-x-y+1,0*y] -2*LISTS['N1']['TRIG']['phi'][6](x,y) -1*LISTS['N1']['TRIG']['phi'][7](x,y)
        LISTS['N1']['TRIG']['phi'][5] = lambda x,y: np.r_[x,x] -1*LISTS['N1']['TRIG']['phi'][6](x,y) +1*LISTS['N1']['TRIG']['phi'][7](x,y)

        LISTS['N1']['TRIG']['curlphi'][0] = lambda x,y: 1 +1*LISTS['N1']['TRIG']['curlphi'][6](x,y) +2*LISTS['N1']['TRIG']['curlphi'][7](x,y)
        LISTS['N1']['TRIG']['curlphi'][1] = lambda x,y: 1 +2*LISTS['N1']['TRIG']['curlphi'][6](x,y) +1*LISTS['N1']['TRIG']['curlphi'][7](x,y)
        LISTS['N1']['TRIG']['curlphi'][2] = lambda x,y: 1 +1*LISTS['N1']['TRIG']['curlphi'][6](x,y) -1*LISTS['N1']['TRIG']['curlphi'][7](x,y)
        LISTS['N1']['TRIG']['curlphi'][3] = lambda x,y: 1 -1*LISTS['N1']['TRIG']['curlphi'][6](x,y) -2*LISTS['N1']['TRIG']['curlphi'][7](x,y)
        LISTS['N1']['TRIG']['curlphi'][4] = lambda x,y: 1 -2*LISTS['N1']['TRIG']['curlphi'][6](x,y) -1*LISTS['N1']['TRIG']['curlphi'][7](x,y)
        LISTS['N1']['TRIG']['curlphi'][5] = lambda x,y: 1 -1*LISTS['N1']['TRIG']['curlphi'][6](x,y) +1*LISTS['N1']['TRIG']['curlphi'][7](x,y)        
        
        LISTS['N1']['TRIG']['LIST_DOF'] = np.c_[2*MESH.TriangleToEdges[:,0]   -1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                2*MESH.TriangleToEdges[:,0]+1 +1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                2*MESH.TriangleToEdges[:,1]   -1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                2*MESH.TriangleToEdges[:,1]+1 +1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                2*MESH.TriangleToEdges[:,2]   -1/2*(MESH.EdgeDirectionTrig[:,2]-1),
                                                2*MESH.TriangleToEdges[:,2]+1 +1/2*(MESH.EdgeDirectionTrig[:,2]-1),
                                                range(2*MESH.NoEdges+0,2*MESH.NoEdges+2*MESH.nt,2),
                                                range(2*MESH.NoEdges+1,2*MESH.NoEdges+2*MESH.nt,2)]
    
        LISTS['N1']['TRIG']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTrig[:,0],
                                                     MESH.EdgeDirectionTrig[:,0],
                                                     MESH.EdgeDirectionTrig[:,1],
                                                     MESH.EdgeDirectionTrig[:,1],
                                                     MESH.EdgeDirectionTrig[:,2],
                                                     MESH.EdgeDirectionTrig[:,2],
                                                     np.ones(MESH.nt),
                                                     np.ones(MESH.nt)]
    ###########################################################################
    
    
    
    
    
    ###########################################################################
    if space == 'N0d':
        
        LISTS['N0d'] = {}
        LISTS['N0d']['TRIG'] = {}
        
        LISTS['N0d']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        LISTS['N0d']['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
        LISTS['N0d']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
        
        LISTS['N0d']['TRIG']['sizeM'] = 3*MESH.nt
        
        LISTS['N0d']['TRIG']['phi'] = {}
        LISTS['N0d']['TRIG']['phi'][0] = lambda x,y: np.r_[-y,x]
        LISTS['N0d']['TRIG']['phi'][1] = lambda x,y: np.r_[-y,x-1]
        LISTS['N0d']['TRIG']['phi'][2] = lambda x,y: np.r_[-y+1,x]
        
        LISTS['N0d']['TRIG']['curlphi'] = {}
        LISTS['N0d']['TRIG']['curlphi'][0] = lambda x,y: 2+0*x
        LISTS['N0d']['TRIG']['curlphi'][1] = lambda x,y: 2+0*x
        LISTS['N0d']['TRIG']['curlphi'][2] = lambda x,y: 2+0*x
        
        LISTS['N0d']['TRIG']['LIST_DOF'] = np.c_[0:3*MESH.nt:3,\
                                                 1:3*MESH.nt:3,\
                                                 2:3*MESH.nt:3]
        LISTS['N0d']['TRIG']['DIRECTION_DOF'] = MESH.EdgeDirectionTrig
    ###########################################################################
    
    ###########################################################################
    if space == 'NC1d':
        
        LISTS['NC1d'] = {}
        LISTS['NC1d']['TRIG'] = {}
        
        LISTS['NC1d']['TRIG']['qp_we_M'] = quadrature.dunavant(order = 2)
        LISTS['NC1d']['TRIG']['qp_we_Mh'] = quadrature.dunavant(order = 1)
        LISTS['NC1d']['TRIG']['qp_we_K'] = quadrature.dunavant(order = 0)
        
        LISTS['NC1d']['TRIG']['sizeM'] = 6*MESH.nt
        
        LISTS['NC1d']['TRIG']['phi'] = {}
        LISTS['NC1d']['TRIG']['phi'][0] = lambda x,y: np.r_[0*x,x]
        LISTS['NC1d']['TRIG']['phi'][1] = lambda x,y: np.r_[-y,0*x]
        LISTS['NC1d']['TRIG']['phi'][2] = lambda x,y: np.r_[-y,-y]
        LISTS['NC1d']['TRIG']['phi'][3] = lambda x,y: np.r_[0*x,x+y-1]
        LISTS['NC1d']['TRIG']['phi'][4] = lambda x,y: np.r_[-x-y+1,0*y]
        LISTS['NC1d']['TRIG']['phi'][5] = lambda x,y: np.r_[x,x]
        
        LISTS['NC1d']['TRIG']['curlphi'] = {}
        LISTS['NC1d']['TRIG']['curlphi'][0] = lambda x,y: 1+0*x
        LISTS['NC1d']['TRIG']['curlphi'][1] = lambda x,y: 1+0*x
        LISTS['NC1d']['TRIG']['curlphi'][2] = lambda x,y: 1+0*x
        LISTS['NC1d']['TRIG']['curlphi'][3] = lambda x,y: 1+0*x
        LISTS['NC1d']['TRIG']['curlphi'][4] = lambda x,y: 1+0*x
        LISTS['NC1d']['TRIG']['curlphi'][5] = lambda x,y: 1+0*x
        
        
        LISTS['NC1d']['TRIG']['LIST_DOF'] = np.c_[2*np.r_[0:3*MESH.nt:3]   -1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                  2*np.r_[0:3*MESH.nt:3]+1 +1/2*(MESH.EdgeDirectionTrig[:,0]-1),
                                                  2*np.r_[1:3*MESH.nt:3]   -1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                  2*np.r_[1:3*MESH.nt:3]+1 +1/2*(MESH.EdgeDirectionTrig[:,1]-1),
                                                  2*np.r_[2:3*MESH.nt:3]   -1/2*(MESH.EdgeDirectionTrig[:,2]-1),
                                                  2*np.r_[2:3*MESH.nt:3]+1 +1/2*(MESH.EdgeDirectionTrig[:,2]-1)].astype(int)
        
        # LISTS['NC1d']['TRIG']['LIST_DOF'] = np.r_[0:6*MESH.nt].reshape(MESH.nt,6)
        
        LISTS['NC1d']['TRIG']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTrig[:,0],
                                                       MESH.EdgeDirectionTrig[:,0],
                                                       MESH.EdgeDirectionTrig[:,1],
                                                       MESH.EdgeDirectionTrig[:,1],
                                                       MESH.EdgeDirectionTrig[:,2],
                                                       MESH.EdgeDirectionTrig[:,2]]

    ###########################################################################
    
    
    
    
    