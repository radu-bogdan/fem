import numpy as npy
from .spaces import spaceInfo
from .. import quadrature

def interp(MESH, space, order, f):
    
    if not space in MESH.FEMLISTS.keys():
        spaceInfo(MESH, space)
        
    p = MESH.p; # t = MESH.t; nt = MESH.nt;
    
    qp,we = quadrature.one_d(order); nqp = len(we)
    
    e0 = MESH.EdgesToVertices[:,1]; e1 = MESH.EdgesToVertices[:,0]
    A0 = p[e1,0]-p[e0,0]; A1 = p[e1,1]-p[e0,1]
    
    scaled_normal = npy.c_[-A1,A0] # scales so we don't need any determinants in the integral below, outstanding, kekw.
    
    phi =  MESH.FEMLISTS[space]['TRIG']['phidual']; lphi = len(phi)
    
    phii = npy.zeros((MESH.NoEdges,lphi))
    ellmatsT = npy.zeros((MESH.NoEdges,lphi))
    
    for i in range(nqp):
        
        qpT_i_1 = A0*qp[i] + p[e0,0]
        qpT_i_2 = A1*qp[i] + p[e0,1]
        
        f_qpT_i = f(qpT_i_1,qpT_i_2)
        
        for j in range(lphi):
            phii[:,j] = phi[j](qp[i])
        
        ellmatsT = ellmatsT + we[i]*npy.sum(f_qpT_i*scaled_normal,axis=1)[:,None]*phii
    
    return ellmatsT.flatten()