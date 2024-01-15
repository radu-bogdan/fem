
from .. import quadrature
import numpy as np

def spaceInfo(MESH,space):
    
    LISTS = MESH.FEMLISTS
    
    lam = {}
    lam[0] = lambda x,y,z : 1-x-y-z
    lam[1] = lambda x,y,z : x
    lam[2] = lambda x,y,z : y
    lam[3] = lambda x,y,z : z
    
    Dlam = {}
    Dlam[0] = lambda x,y,z : np.r_[-1,-1,-1]
    Dlam[1] = lambda x,y,z : np.r_[ 1, 0, 0]
    Dlam[2] = lambda x,y,z : np.r_[ 0, 1, 0]
    Dlam[3] = lambda x,y,z : np.r_[ 0, 0, 1]
    
    # Dlam0 x Dlam1 = ( 0,-1, 1)
    # Dlam0 x Dlam2 = ( 1, 0,-1)
    # Dlam0 x Dlam3 = (-1, 1, 0)
    # Dlam1 x Dlam2 = ( 0, 0, 1)
    # Dlam1 x Dlam3 = ( 0,-1, 0)
    # Dlam2 x Dlam3 = ( 1, 0, 0)
    
    W2 = lambda i,j,k,x,y,z : lam[i](x,y,z)*np.cross(Dlam[j](x,y,z),Dlam[k](x,y,z))
    divW2 = lambda i,j,k,x,y,z : np.dot(Dlam[i](x,y,z),np.cross(Dlam[j](x,y,z),Dlam[k](x,y,z)))
    
    W = lambda i,j,k,x,y,z : W2(i,j,k,x,y,z)+\
                             W2(j,k,i,x,y,z)+\
                             W2(k,i,j,x,y,z)
                             
    divW = lambda i,j,k,x,y,z : divW2(i,j,k,x,y,z)+\
                                divW2(j,k,i,x,y,z)+\
                                divW2(k,i,j,x,y,z)
    
    ###########################################################################
    if space == 'RT0':
        
        LISTS['RT0'] = {}
        LISTS['RT0']['TET'] = {}
        
        LISTS['RT0']['TET']['qp_we_M']  = quadrature.keast(order = 2)
        LISTS['RT0']['TET']['qp_we_K']  = quadrature.keast(order = 0)
        
        LISTS['RT0']['TET']['sizeM'] = MESH.NoFaces
        
        LISTS['RT0']['TET']['phi'] = {}
        LISTS['RT0']['TET']['phi'][0] = lambda x,y,z: W(1,2,3,x,y,z)
        LISTS['RT0']['TET']['phi'][1] = lambda x,y,z: -W(0,2,3,x,y,z)
        LISTS['RT0']['TET']['phi'][2] = lambda x,y,z: W(0,1,3,x,y,z)
        LISTS['RT0']['TET']['phi'][3] = lambda x,y,z: -W(0,1,2,x,y,z)
        
        LISTS['RT0']['TET']['divphi'] = {}
        LISTS['RT0']['TET']['divphi'][0] = lambda x,y,z: divW(1,2,3,x,y,z)
        LISTS['RT0']['TET']['divphi'][1] = lambda x,y,z: -divW(0,2,3,x,y,z)
        LISTS['RT0']['TET']['divphi'][2] = lambda x,y,z: divW(0,1,3,x,y,z)
        LISTS['RT0']['TET']['divphi'][3] = lambda x,y,z: -divW(0,1,2,x,y,z)
        
        LISTS['RT0']['TET']['LIST_DOF'] = MESH.TetsToFaces
        LISTS['RT0']['TET']['DIRECTION_DOF'] = MESH.DirectionFaces
        
        LISTS['RT0']['B'] = {}
        LISTS['RT0']['B']['phi'] = {}
        LISTS['RT0']['B']['phi'][0] = lambda x,y: 1 #?
        LISTS['RT0']['B']['qp_we_B'] = quadrature.one_d(order = 0)
        
        LISTS['RT0']['B']['LIST_DOF'] = MESH.Boundary_Faces
        
        # LISTS['N0']['B']['LIST_DOF_E1'] = MESH.IntEdgesToTriangles[:,0]
        # LISTS['N0']['B']['LIST_DOF_E1'] = MESH.IntEdgesToTriangles[:,1]
        
        # LISTS['N0']['TET']['phidual'] = {}
        # LISTS['N0']['TET']['phidual'][0] = lambda x: 2+0*x
    ###########################################################################

    
    # ###########################################################################
    # if space == 'EJ1':
        
    #     LISTS['EJ1'] = {}
    #     LISTS['EJ1']['TET'] = {}
        
    #     LISTS['EJ1']['TET']['qp_we_M'] = quadrature.dunavant(order = 4)
    #     LISTS['EJ1']['TET']['qp_we_Mh'] = quadrature.dunavant(order = '2m')
    #     LISTS['EJ1']['TET']['qp_we_K'] = quadrature.dunavant(order = 2)
        
    #     LISTS['EJ1']['sizeM'] = MESH.NoEdges + 3*MESH.nt
        
    #     LISTS['EJ1']['TET']['phi'] = {}        
    #     LISTS['EJ1']['TET']['phi'][0] = lambda x,y: np.r_[-y,x]
    #     LISTS['EJ1']['TET']['phi'][1] = lambda x,y: np.r_[-y,x-1]
    #     LISTS['EJ1']['TET']['phi'][2] = lambda x,y: np.r_[-y+1,x]
        
    #     LISTS['EJ1']['TET']['curlphi'] = {}   
    #     LISTS['EJ1']['TET']['curlphi'][0] = lambda x,y: 2+0*x # RT0 basis
    #     LISTS['EJ1']['TET']['curlphi'][1] = lambda x,y: 2+0*x # RT0 basis
    #     LISTS['EJ1']['TET']['curlphi'][2] = lambda x,y: 2+0*x # RT0 basis
        
    #     LISTS['EJ1']['TET']['phi'][3] = lambda x,y: np.r_[4*x*y,4*x*y]
    #     LISTS['EJ1']['TET']['phi'][4] = lambda x,y: np.r_[-4*y*(1-x-y),0*y]
    #     LISTS['EJ1']['TET']['phi'][5] = lambda x,y: np.r_[0*x,4*x*(1-x-y)]
        
    #     LISTS['EJ1']['TET']['curlphi'][3] = lambda x,y: 4*y-4*x
    #     LISTS['EJ1']['TET']['curlphi'][4] = lambda x,y: 4-4*x-8*y
    #     LISTS['EJ1']['TET']['curlphi'][5] = lambda x,y: 4-8*x-4*y
        
    #     LISTS['EJ1']['TET']['phi'][0] = lambda x,y: LISTS['EJ1']['TET']['phi'][0](x,y) -1/2*LISTS['EJ1']['TET']['phi'][4](x,y) -1/2*LISTS['EJ1']['TET']['phi'][5](x,y)
    #     LISTS['EJ1']['TET']['phi'][1] = lambda x,y: LISTS['EJ1']['TET']['phi'][1](x,y) +1/2*LISTS['EJ1']['TET']['phi'][5](x,y) +1/2*LISTS['EJ1']['TET']['phi'][3](x,y)
    #     LISTS['EJ1']['TET']['phi'][2] = lambda x,y: LISTS['EJ1']['TET']['phi'][2](x,y) -1/2*LISTS['EJ1']['TET']['phi'][3](x,y) +1/2*LISTS['EJ1']['TET']['phi'][4](x,y)
        
    #     LISTS['EJ1']['TET']['curlphi'][0] = lambda x,y: LISTS['EJ1']['TET']['curlphi'][0](x,y) -1/2*LISTS['EJ1']['TET']['curlphi'][4](x,y) -1/2*LISTS['EJ1']['TET']['curlphi'][5](x,y)
    #     LISTS['EJ1']['TET']['curlphi'][1] = lambda x,y: LISTS['EJ1']['TET']['curlphi'][1](x,y) +1/2*LISTS['EJ1']['TET']['curlphi'][5](x,y) +1/2*LISTS['EJ1']['TET']['curlphi'][3](x,y)
    #     LISTS['EJ1']['TET']['curlphi'][2] = lambda x,y: LISTS['EJ1']['TET']['curlphi'][2](x,y) -1/2*LISTS['EJ1']['TET']['curlphi'][3](x,y) +1/2*LISTS['EJ1']['TET']['curlphi'][4](x,y)       
        
        
    #     LISTS['EJ1']['TET']['LIST_DOF'] = np.c_[MESH.TriangleToEdges,
    #                                               range(MESH.NoEdges+0,MESH.NoEdges+3*MESH.nt,3),
    #                                               range(MESH.NoEdges+1,MESH.NoEdges+3*MESH.nt,3),
    #                                               range(MESH.NoEdges+2,MESH.NoEdges+3*MESH.nt,3)]
        
    #     LISTS['EJ1']['TET']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTET,
    #                                                   np.ones(MESH.nt),
    #                                                   np.ones(MESH.nt),
    #                                                   np.ones(MESH.nt)]

    #     LISTS['EJ1']['B'] = {}
    #     LISTS['EJ1']['B']['phi'] = {}
    #     LISTS['EJ1']['B']['phi'][0] = lambda x: 1
    #     LISTS['EJ1']['B']['qp_we_B'] = quadrature.one_d(order = 0)
        
    #     LISTS['EJ1']['B']['LIST_DOF'] = MESH.Boundary_Edges
    # ###########################################################################
    
    
    # ###########################################################################
    # if space == 'NC1':
        
    #     LISTS['NC1'] = {}
    #     LISTS['NC1']['TET'] = {}
        
    #     LISTS['NC1']['TET']['qp_we_M'] = quadrature.dunavant(order = 2)
    #     LISTS['NC1']['TET']['qp_we_Mh'] = quadrature.dunavant(order = 1)
    #     LISTS['NC1']['TET']['qp_we_K'] = quadrature.dunavant(order = 0)
        
    #     LISTS['NC1']['TET']['sizeM'] = 2*MESH.NoEdges
        
    #     LISTS['NC1']['TET']['phi'] = {}
    #     LISTS['NC1']['TET']['phi'][0] = lambda x,y: np.r_[0*x,x]
    #     LISTS['NC1']['TET']['phi'][1] = lambda x,y: np.r_[-y,0*x]
    #     LISTS['NC1']['TET']['phi'][2] = lambda x,y: np.r_[-y,-y]
    #     LISTS['NC1']['TET']['phi'][3] = lambda x,y: np.r_[0*x,x+y-1]
    #     LISTS['NC1']['TET']['phi'][4] = lambda x,y: np.r_[-x-y+1,0*y]
    #     LISTS['NC1']['TET']['phi'][5] = lambda x,y: np.r_[x,x]
        
    #     LISTS['NC1']['TET']['curlphi'] = {}
    #     LISTS['NC1']['TET']['curlphi'][0] = lambda x,y: 1+0*x
    #     LISTS['NC1']['TET']['curlphi'][1] = lambda x,y: 1+0*x
    #     LISTS['NC1']['TET']['curlphi'][2] = lambda x,y: 1+0*x
    #     LISTS['NC1']['TET']['curlphi'][3] = lambda x,y: 1+0*x
    #     LISTS['NC1']['TET']['curlphi'][4] = lambda x,y: 1+0*x
    #     LISTS['NC1']['TET']['curlphi'][5] = lambda x,y: 1+0*x
        
    #     LISTS['NC1']['TET']['LIST_DOF'] = np.c_[2*MESH.TriangleToEdges[:,0]   -1/2*(MESH.EdgeDirectionTET[:,0]-1),
    #                                              2*MESH.TriangleToEdges[:,0]+1 +1/2*(MESH.EdgeDirectionTET[:,0]-1),
    #                                              2*MESH.TriangleToEdges[:,1]   -1/2*(MESH.EdgeDirectionTET[:,1]-1),
    #                                              2*MESH.TriangleToEdges[:,1]+1 +1/2*(MESH.EdgeDirectionTET[:,1]-1),
    #                                              2*MESH.TriangleToEdges[:,2]   -1/2*(MESH.EdgeDirectionTET[:,2]-1),
    #                                              2*MESH.TriangleToEdges[:,2]+1 +1/2*(MESH.EdgeDirectionTET[:,2]-1)]
    
    #     LISTS['NC1']['TET']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTET[:,0],
    #                                                   MESH.EdgeDirectionTET[:,0],
    #                                                   MESH.EdgeDirectionTET[:,1],
    #                                                   MESH.EdgeDirectionTET[:,1],
    #                                                   MESH.EdgeDirectionTET[:,2],
    #                                                   MESH.EdgeDirectionTET[:,2]]
        
    #     LISTS['NC1']['B'] = {}
    #     LISTS['NC1']['B']['phi'] = {}
    #     LISTS['NC1']['B']['phi'][0] = lambda x: 1-x
    #     LISTS['NC1']['B']['phi'][1] = lambda x: x
    #     LISTS['NC1']['B']['qp_we_B'] = quadrature.one_d(order = 2)
        
    #     LISTS['NC1']['B']['LIST_DOF'] = np.c_[2*MESH.Boundary_Edges,
    #                                           2*MESH.Boundary_Edges + 1]
        
    #     LISTS['NC1']['B']['LIST_DOF_E1'] = np.c_[2*MESH.IntEdgesToTriangles[:,0],
    #                                              2*MESH.IntEdgesToTriangles[:,0]+1]
    #     LISTS['NC1']['B']['LIST_DOF_E1'] = np.c_[2*MESH.IntEdgesToTriangles[:,1],
    #                                              2*MESH.IntEdgesToTriangles[:,1]+1]

                                              
    #     LISTS['NC1']['B']['LIST_DOF_E'] = np.c_[2*MESH.NonSingle_Edges,
    #                                             2*MESH.NonSingle_Edges + 1]
        
    #     LISTS['NC1']['TET']['phidual'] = {}
    #     LISTS['NC1']['TET']['phidual'][0] = lambda x: 6*x-2 # dual to x and 1-x on the edge...
    #     LISTS['NC1']['TET']['phidual'][1] = lambda x: -6*x+4
    # ###########################################################################
    
    
    
    # ###########################################################################
    # if space == 'N1':
        
    #     LISTS['N1'] = {}
    #     LISTS['N1']['TET'] = {}
        
    #     LISTS['N1']['TET']['qp_we_M'] = quadrature.dunavant(order = 4)
    #     LISTS['N1']['TET']['qp_we_Mh'] = quadrature.dunavant(order = '2l')
    #     LISTS['N1']['TET']['qp_we_K'] = quadrature.dunavant(order = 2)
        
    #     LISTS['N1']['TET']['sizeM'] = 2*MESH.NoEdges + 2*MESH.nt
        
    #     LISTS['N1']['TET']['phi'] = {}
    #     LISTS['N1']['TET']['phi'][6] = lambda x,y: np.r_[-y*(y-1),x*y]
    #     LISTS['N1']['TET']['phi'][7] = lambda x,y: np.r_[-x*y,x*(x-1)]

    #     LISTS['N1']['TET']['curlphi'] = {}
    #     LISTS['N1']['TET']['curlphi'][6] = lambda x,y: 3*y-1
    #     LISTS['N1']['TET']['curlphi'][7] = lambda x,y: 3*x-1

    #     LISTS['N1']['TET']['phi'][0] = lambda x,y: np.r_[0*x,x] +1*LISTS['N1']['TET']['phi'][6](x,y) +2*LISTS['N1']['TET']['phi'][7](x,y)
    #     LISTS['N1']['TET']['phi'][1] = lambda x,y: np.r_[-y,0*x] +2*LISTS['N1']['TET']['phi'][6](x,y) +1*LISTS['N1']['TET']['phi'][7](x,y)
    #     LISTS['N1']['TET']['phi'][2] = lambda x,y: np.r_[-y,-y] +1*LISTS['N1']['TET']['phi'][6](x,y) -1*LISTS['N1']['TET']['phi'][7](x,y)
    #     LISTS['N1']['TET']['phi'][3] = lambda x,y: np.r_[0*x,x+y-1] -1*LISTS['N1']['TET']['phi'][6](x,y) -2*LISTS['N1']['TET']['phi'][7](x,y)
    #     LISTS['N1']['TET']['phi'][4] = lambda x,y: np.r_[-x-y+1,0*y] -2*LISTS['N1']['TET']['phi'][6](x,y) -1*LISTS['N1']['TET']['phi'][7](x,y)
    #     LISTS['N1']['TET']['phi'][5] = lambda x,y: np.r_[x,x] -1*LISTS['N1']['TET']['phi'][6](x,y) +1*LISTS['N1']['TET']['phi'][7](x,y)

    #     LISTS['N1']['TET']['curlphi'][0] = lambda x,y: 1 +1*LISTS['N1']['TET']['curlphi'][6](x,y) +2*LISTS['N1']['TET']['curlphi'][7](x,y)
    #     LISTS['N1']['TET']['curlphi'][1] = lambda x,y: 1 +2*LISTS['N1']['TET']['curlphi'][6](x,y) +1*LISTS['N1']['TET']['curlphi'][7](x,y)
    #     LISTS['N1']['TET']['curlphi'][2] = lambda x,y: 1 +1*LISTS['N1']['TET']['curlphi'][6](x,y) -1*LISTS['N1']['TET']['curlphi'][7](x,y)
    #     LISTS['N1']['TET']['curlphi'][3] = lambda x,y: 1 -1*LISTS['N1']['TET']['curlphi'][6](x,y) -2*LISTS['N1']['TET']['curlphi'][7](x,y)
    #     LISTS['N1']['TET']['curlphi'][4] = lambda x,y: 1 -2*LISTS['N1']['TET']['curlphi'][6](x,y) -1*LISTS['N1']['TET']['curlphi'][7](x,y)
    #     LISTS['N1']['TET']['curlphi'][5] = lambda x,y: 1 -1*LISTS['N1']['TET']['curlphi'][6](x,y) +1*LISTS['N1']['TET']['curlphi'][7](x,y)        
        
    #     LISTS['N1']['TET']['LIST_DOF'] = np.c_[2*MESH.TriangleToEdges[:,0]   -1/2*(MESH.EdgeDirectionTET[:,0]-1),
    #                                             2*MESH.TriangleToEdges[:,0]+1 +1/2*(MESH.EdgeDirectionTET[:,0]-1),
    #                                             2*MESH.TriangleToEdges[:,1]   -1/2*(MESH.EdgeDirectionTET[:,1]-1),
    #                                             2*MESH.TriangleToEdges[:,1]+1 +1/2*(MESH.EdgeDirectionTET[:,1]-1),
    #                                             2*MESH.TriangleToEdges[:,2]   -1/2*(MESH.EdgeDirectionTET[:,2]-1),
    #                                             2*MESH.TriangleToEdges[:,2]+1 +1/2*(MESH.EdgeDirectionTET[:,2]-1),
    #                                             range(2*MESH.NoEdges+0,2*MESH.NoEdges+2*MESH.nt,2),
    #                                             range(2*MESH.NoEdges+1,2*MESH.NoEdges+2*MESH.nt,2)]
    
    #     LISTS['N1']['TET']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTET[:,0],
    #                                                  MESH.EdgeDirectionTET[:,0],
    #                                                  MESH.EdgeDirectionTET[:,1],
    #                                                  MESH.EdgeDirectionTET[:,1],
    #                                                  MESH.EdgeDirectionTET[:,2],
    #                                                  MESH.EdgeDirectionTET[:,2],
    #                                                  np.ones(MESH.nt),
    #                                                  np.ones(MESH.nt)]
        
    #     LISTS['N1']['B'] = {}
    #     LISTS['N1']['B']['LIST_DOF'] = np.c_[2*MESH.Boundary_Edges,
    #                                          2*MESH.Boundary_Edges+1]
                                             
    # ###########################################################################
    
    
    
    
    
    # ###########################################################################
    # if space == 'N0d':
        
    #     LISTS['N0d'] = {}
    #     LISTS['N0d']['TET'] = {}
        
    #     LISTS['N0d']['TET']['qp_we_M'] = quadrature.dunavant(order = 2)
    #     LISTS['N0d']['TET']['qp_we_Mh'] = quadrature.dunavant(order = 1)
    #     LISTS['N0d']['TET']['qp_we_K'] = quadrature.dunavant(order = 0)
        
    #     LISTS['N0d']['TET']['sizeM'] = 3*MESH.nt
        
    #     LISTS['N0d']['TET']['phi'] = {}
    #     LISTS['N0d']['TET']['phi'][0] = lambda x,y: np.r_[-y,x]
    #     LISTS['N0d']['TET']['phi'][1] = lambda x,y: np.r_[-y,x-1]
    #     LISTS['N0d']['TET']['phi'][2] = lambda x,y: np.r_[-y+1,x]
        
    #     LISTS['N0d']['TET']['curlphi'] = {}
    #     LISTS['N0d']['TET']['curlphi'][0] = lambda x,y: 2+0*x
    #     LISTS['N0d']['TET']['curlphi'][1] = lambda x,y: 2+0*x
    #     LISTS['N0d']['TET']['curlphi'][2] = lambda x,y: 2+0*x
        
    #     LISTS['N0d']['TET']['LIST_DOF'] = np.c_[0:3*MESH.nt:3,\
    #                                              1:3*MESH.nt:3,\
    #                                              2:3*MESH.nt:3]
        
    #     LISTS['N0d']['TET']['LIST_DOF_C'] = np.c_[0:3*MESH.nt:3,\
    #                                                1:3*MESH.nt:3,\
    #                                                2:3*MESH.nt:3]
            
    #     LISTS['N0d']['TET']['DIRECTION_DOF'] = MESH.EdgeDirectionTET
    # ###########################################################################
    
    # ###########################################################################
    # if space == 'NC1d':
        
    #     LISTS['NC1d'] = {}
    #     LISTS['NC1d']['TET'] = {}
        
    #     LISTS['NC1d']['TET']['qp_we_M'] = quadrature.dunavant(order = 2)
    #     LISTS['NC1d']['TET']['qp_we_Mh'] = quadrature.dunavant(order = 1)
    #     LISTS['NC1d']['TET']['qp_we_K'] = quadrature.dunavant(order = 0)
        
    #     LISTS['NC1d']['TET']['sizeM'] = 6*MESH.nt
        
    #     LISTS['NC1d']['TET']['phi'] = {}
    #     LISTS['NC1d']['TET']['phi'][0] = lambda x,y: np.r_[0*x,x]
    #     LISTS['NC1d']['TET']['phi'][1] = lambda x,y: np.r_[-y,0*x]
    #     LISTS['NC1d']['TET']['phi'][2] = lambda x,y: np.r_[-y,-y]
    #     LISTS['NC1d']['TET']['phi'][3] = lambda x,y: np.r_[0*x,x+y-1]
    #     LISTS['NC1d']['TET']['phi'][4] = lambda x,y: np.r_[-x-y+1,0*y]
    #     LISTS['NC1d']['TET']['phi'][5] = lambda x,y: np.r_[x,x]
        
    #     LISTS['NC1d']['TET']['curlphi'] = {}
    #     LISTS['NC1d']['TET']['curlphi'][0] = lambda x,y: 1+0*x
    #     LISTS['NC1d']['TET']['curlphi'][1] = lambda x,y: 1+0*x
    #     LISTS['NC1d']['TET']['curlphi'][2] = lambda x,y: 1+0*x
    #     LISTS['NC1d']['TET']['curlphi'][3] = lambda x,y: 1+0*x
    #     LISTS['NC1d']['TET']['curlphi'][4] = lambda x,y: 1+0*x
    #     LISTS['NC1d']['TET']['curlphi'][5] = lambda x,y: 1+0*x
        
    #     LISTS['NC1d']['TET']['LIST_DOF'] = np.r_[0:6*MESH.nt].reshape(MESH.nt,6)
        
    #     LISTS['NC1d']['TET']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTET[:,0],
    #                                                    MESH.EdgeDirectionTET[:,0],
    #                                                    MESH.EdgeDirectionTET[:,1],
    #                                                    MESH.EdgeDirectionTET[:,1],
    #                                                    MESH.EdgeDirectionTET[:,2],
    #                                                    MESH.EdgeDirectionTET[:,2]]

    # ###########################################################################
    # if space == 'N1d':
        
    #     LISTS['N1d'] = {}
    #     LISTS['N1d']['TET'] = {}
        
    #     LISTS['N1d']['TET']['qp_we_M'] = quadrature.dunavant(order = 4)
    #     LISTS['N1d']['TET']['qp_we_Mh'] = quadrature.dunavant(order = '2l')
    #     LISTS['N1d']['TET']['qp_we_K'] = quadrature.dunavant(order = 2)
        
    #     LISTS['N1d']['TET']['sizeM'] = 8*MESH.nt
        
    #     LISTS['N1d']['TET']['phi'] = {}
    #     LISTS['N1d']['TET']['phi'][6] = lambda x,y: np.r_[-y*(y-1),x*y]
    #     LISTS['N1d']['TET']['phi'][7] = lambda x,y: np.r_[-x*y,x*(x-1)]

    #     LISTS['N1d']['TET']['curlphi'] = {}
    #     LISTS['N1d']['TET']['curlphi'][6] = lambda x,y: 3*y-1
    #     LISTS['N1d']['TET']['curlphi'][7] = lambda x,y: 3*x-1

    #     LISTS['N1d']['TET']['phi'][0] = lambda x,y: np.r_[0*x,x] +1*LISTS['N1d']['TET']['phi'][6](x,y) +2*LISTS['N1d']['TET']['phi'][7](x,y)
    #     LISTS['N1d']['TET']['phi'][1] = lambda x,y: np.r_[-y,0*x] +2*LISTS['N1d']['TET']['phi'][6](x,y) +1*LISTS['N1d']['TET']['phi'][7](x,y)
    #     LISTS['N1d']['TET']['phi'][2] = lambda x,y: np.r_[-y,-y] +1*LISTS['N1d']['TET']['phi'][6](x,y) -1*LISTS['N1d']['TET']['phi'][7](x,y)
    #     LISTS['N1d']['TET']['phi'][3] = lambda x,y: np.r_[0*x,x+y-1] -1*LISTS['N1d']['TET']['phi'][6](x,y) -2*LISTS['N1d']['TET']['phi'][7](x,y)
    #     LISTS['N1d']['TET']['phi'][4] = lambda x,y: np.r_[-x-y+1,0*y] -2*LISTS['N1d']['TET']['phi'][6](x,y) -1*LISTS['N1d']['TET']['phi'][7](x,y)
    #     LISTS['N1d']['TET']['phi'][5] = lambda x,y: np.r_[x,x] -1*LISTS['N1d']['TET']['phi'][6](x,y) +1*LISTS['N1d']['TET']['phi'][7](x,y)

    #     LISTS['N1d']['TET']['curlphi'][0] = lambda x,y: 1 +1*LISTS['N1d']['TET']['curlphi'][6](x,y) +2*LISTS['N1d']['TET']['curlphi'][7](x,y)
    #     LISTS['N1d']['TET']['curlphi'][1] = lambda x,y: 1 +2*LISTS['N1d']['TET']['curlphi'][6](x,y) +1*LISTS['N1d']['TET']['curlphi'][7](x,y)
    #     LISTS['N1d']['TET']['curlphi'][2] = lambda x,y: 1 +1*LISTS['N1d']['TET']['curlphi'][6](x,y) -1*LISTS['N1d']['TET']['curlphi'][7](x,y)
    #     LISTS['N1d']['TET']['curlphi'][3] = lambda x,y: 1 -1*LISTS['N1d']['TET']['curlphi'][6](x,y) -2*LISTS['N1d']['TET']['curlphi'][7](x,y)
    #     LISTS['N1d']['TET']['curlphi'][4] = lambda x,y: 1 -2*LISTS['N1d']['TET']['curlphi'][6](x,y) -1*LISTS['N1d']['TET']['curlphi'][7](x,y)
    #     LISTS['N1d']['TET']['curlphi'][5] = lambda x,y: 1 -1*LISTS['N1d']['TET']['curlphi'][6](x,y) +1*LISTS['N1d']['TET']['curlphi'][7](x,y)        
        
    #     LISTS['N1d']['TET']['LIST_DOF'] = np.r_[0:8*MESH.nt].reshape(MESH.nt,8)
    
    #     LISTS['N1d']['TET']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTET[:,0],
    #                                                   MESH.EdgeDirectionTET[:,0],
    #                                                   MESH.EdgeDirectionTET[:,1],
    #                                                   MESH.EdgeDirectionTET[:,1],
    #                                                   MESH.EdgeDirectionTET[:,2],
    #                                                   MESH.EdgeDirectionTET[:,2],
    #                                                   np.ones(MESH.nt),
    #                                                   np.ones(MESH.nt)]
    # ###########################################################################
    
    
    
    