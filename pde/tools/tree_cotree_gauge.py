from .mst import *
import numpy as np
import scipy.sparse as sp

def tree_cotree_gauge(MESH, random_edges=True):
    if random_edges:
        random = np.random.permutation(MESH.EdgesToVertices[:,:2].shape[0])
        newListOfEdges = MESH.EdgesToVertices[random,:2]
    else:
        newListOfEdges = MESH.EdgesToVertices[:,:2]

    g = Graph(MESH.np)

    for i in range(newListOfEdges.shape[0]):
        g.addEdge(newListOfEdges[i,0],newListOfEdges[i,1],i)

    g.KruskalMST()
    indices = np.array(g.MST)[:,2]

    if random_edges:
        LIST_DOF = np.setdiff1d(np.r_[:MESH.NoEdges],random[indices])
    else:
        LIST_DOF = np.setdiff1d(np.r_[:MESH.NoEdges],indices)

    DD = sp.eye(MESH.NoEdges, format = 'csc')

    R = DD[:,LIST_DOF]
    return R