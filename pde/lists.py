import numpy as np

def lists(MESH):
    LISTS = {}
    LISTS['RT0'] = {}
    LISTS['EJ1'] = {}
    LISTS['BDM1'] = {}
    LISTS['RT1'] = {}
    LISTS['BDFM1'] = {}
    LISTS['P0'] = {}
    LISTS['P1'] = {}
    LISTS['P1d'] = {}
    LISTS['P2'] = {}
    LISTS['Q0'] = {}
    LISTS['Q1'] = {}
    LISTS['Q1d'] = {}
    LISTS['Q2'] = {}

    #############################################################################################################
    # RT0 trig
    #############################################################################################################
    LISTS['RT0']['TRIG'] = {}
    LISTS['RT0']['TRIG']['LIST_DOF'] = MESH.TriangleToEdges
    LISTS['RT0']['TRIG']['DIRECTION_DOF'] = MESH.EdgeDirectionTrig
    #############################################################################################################


    #############################################################################################################
    # RT0 quad
    #############################################################################################################
    LISTS['RT0']['QUAD'] = {}
    LISTS['RT0']['QUAD']['LIST_DOF'] = MESH.QuadToEdges
    LISTS['RT0']['QUAD']['DIRECTION_DOF'] = MESH.EdgeDirectionQuad
    #############################################################################################################


    #############################################################################################################
    # EJ1 trig
    #############################################################################################################
    LISTS['EJ1']['TRIG'] = {}
    LISTS['EJ1']['TRIG']['LIST_DOF'] = np.c_[MESH.TriangleToEdges,
                                             range(MESH.NoEdges+0,MESH.NoEdges+3*MESH.nt,3),
                                             range(MESH.NoEdges+1,MESH.NoEdges+3*MESH.nt,3),
                                             range(MESH.NoEdges+2,MESH.NoEdges+3*MESH.nt,3)]

    LISTS['EJ1']['TRIG']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionTrig,
                                                  np.ones(MESH.nt),
                                                  np.ones(MESH.nt),
                                                  np.ones(MESH.nt)]
    #############################################################################################################


    #############################################################################################################
    # BDM1 trig
    #############################################################################################################
    LISTS['BDM1']['TRIG'] = {}
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
    #############################################################################################################


    #############################################################################################################
    # BDM1 quad
    #############################################################################################################
    LISTS['BDM1']['QUAD'] = {}
    LISTS['BDM1']['QUAD']['LIST_DOF'] = np.c_[2*MESH.QuadToEdges[:,0]   -1/2*(MESH.EdgeDirectionQuad[:,0]-1),
                                              2*MESH.QuadToEdges[:,0]+1 +1/2*(MESH.EdgeDirectionQuad[:,0]-1),
                                              2*MESH.QuadToEdges[:,1]   -1/2*(MESH.EdgeDirectionQuad[:,1]-1),
                                              2*MESH.QuadToEdges[:,1]+1 +1/2*(MESH.EdgeDirectionQuad[:,1]-1),
                                              2*MESH.QuadToEdges[:,2]   -1/2*(MESH.EdgeDirectionQuad[:,2]-1),
                                              2*MESH.QuadToEdges[:,2]+1 +1/2*(MESH.EdgeDirectionQuad[:,2]-1),
                                              2*MESH.QuadToEdges[:,3]   -1/2*(MESH.EdgeDirectionQuad[:,3]-1),
                                              2*MESH.QuadToEdges[:,3]+1 +1/2*(MESH.EdgeDirectionQuad[:,3]-1)]

    LISTS['BDM1']['QUAD']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionQuad[:,0],
                                                   MESH.EdgeDirectionQuad[:,0],
                                                   MESH.EdgeDirectionQuad[:,1],
                                                   MESH.EdgeDirectionQuad[:,1],
                                                   MESH.EdgeDirectionQuad[:,2],
                                                   MESH.EdgeDirectionQuad[:,2],
                                                   MESH.EdgeDirectionQuad[:,3],
                                                   MESH.EdgeDirectionQuad[:,3]]
    #############################################################################################################


    #############################################################################################################
    # RT1 trig
    #############################################################################################################
    LISTS['RT1']['TRIG'] = {}
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
    #############################################################################################################


    #############################################################################################################
    # RT1 quad
    #############################################################################################################
    LISTS['BDFM1']['QUAD'] = {}
    LISTS['BDFM1']['QUAD']['LIST_DOF'] = np.c_[2*MESH.QuadToEdges[:,0]   -1/2*(MESH.EdgeDirectionQuad[:,0]-1),
                                               2*MESH.QuadToEdges[:,0]+1 +1/2*(MESH.EdgeDirectionQuad[:,0]-1),
                                               2*MESH.QuadToEdges[:,1]   -1/2*(MESH.EdgeDirectionQuad[:,1]-1),
                                               2*MESH.QuadToEdges[:,1]+1 +1/2*(MESH.EdgeDirectionQuad[:,1]-1),
                                               2*MESH.QuadToEdges[:,2]   -1/2*(MESH.EdgeDirectionQuad[:,2]-1),
                                               2*MESH.QuadToEdges[:,2]+1 +1/2*(MESH.EdgeDirectionQuad[:,2]-1),
                                               2*MESH.QuadToEdges[:,3]   -1/2*(MESH.EdgeDirectionQuad[:,3]-1),
                                               2*MESH.QuadToEdges[:,3]+1 +1/2*(MESH.EdgeDirectionQuad[:,3]-1),
                                               range(2*MESH.NoEdges+2*MESH.nt+0,2*MESH.NoEdges+2*MESH.nt+2*MESH.nq,2),
                                               range(2*MESH.NoEdges+2*MESH.nt+1,2*MESH.NoEdges+2*MESH.nt+2*MESH.nq,2)]

    LISTS['BDFM1']['QUAD']['DIRECTION_DOF'] = np.c_[MESH.EdgeDirectionQuad[:,0],
                                                    MESH.EdgeDirectionQuad[:,0],
                                                    MESH.EdgeDirectionQuad[:,1],
                                                    MESH.EdgeDirectionQuad[:,1],
                                                    MESH.EdgeDirectionQuad[:,2],
                                                    MESH.EdgeDirectionQuad[:,2],
                                                    MESH.EdgeDirectionQuad[:,3],
                                                    MESH.EdgeDirectionQuad[:,3],
                                                    np.ones(MESH.nq),
                                                    np.ones(MESH.nq)]
    #############################################################################################################


    #############################################################################################################
    # Pk
    #############################################################################################################
   
    LISTS['P0']['TRIG'] = {}
    LISTS['P0']['TRIG']['LIST_DOF'] = np.r_[0:MESH.nt][:,None]

    LISTS['P1']['TRIG'] = {}
    LISTS['P1']['TRIG']['LIST_DOF'] = MESH.t
    
    LISTS['P1']['B'] = {}
    LISTS['P1']['B']['LIST_DOF'] = MESH.e
    
    # LISTS['P2']['B'] = {}
    # LISTS['P2']['B']['LIST_DOF'] = MESH.e

    LISTS['P1d']['TRIG'] = {}
    LISTS['P1d']['TRIG']['LIST_DOF'] = np.r_[0:3*MESH.nt].reshape(MESH.nt,3)

    LISTS['P2']['TRIG'] = {}
    LISTS['P2']['TRIG']['LIST_DOF'] = np.c_[MESH.t,MESH.np+MESH.TriangleToEdges].astype(np.uint64)
    
    LISTS['P2']['B'] = {}
    LISTS['P2']['B']['LIST_DOF'] = np.c_[MESH.e,MESH.np+MESH.Boundary.Edges].astype(np.uint64)


    LISTS['Q0']['QUAD'] = {}
    LISTS['Q0']['QUAD']['LIST_DOF'] = np.r_[0:MESH.nq]

    LISTS['Q1']['QUAD'] = {}
    LISTS['Q1']['QUAD']['LIST_DOF'] = MESH.q

    LISTS['P1d']['QUAD'] = {}
    LISTS['P1d']['QUAD']['LIST_DOF'] =  np.r_[0:3*MESH.nq].reshape(MESH.nq,3)

    LISTS['Q1d']['QUAD'] = {}
    LISTS['Q1d']['QUAD']['LIST_DOF'] =  np.r_[0:4*MESH.nq].reshape(MESH.nq,4)

    LISTS['P2']['QUAD'] = {}
    LISTS['P2']['QUAD']['LIST_DOF'] = np.c_[MESH.q,MESH.np+MESH.QuadToEdges]

    return LISTS
    
# if __name__ == "__main__":
#     LISTS = LISTS()







