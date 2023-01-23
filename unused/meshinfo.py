#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 23:16:29 2022

@author: bogdan
"""

# import gmsh
import numpy
from sympy import npartitions
numpy.set_printoptions(edgeitems=30, linewidth = 1000000)
import matplotlib.pyplot as plt
# from IPython.core.debugger import set_trace
    
def make_mesh_data_hybrid(p,e,t,q):
    MESH = {}
    if t.size != 0:
        edges_trigs = numpy.r_[numpy.c_[t[:,1],t[:,2]],
                               numpy.c_[t[:,2],t[:,0]],
                               numpy.c_[t[:,0],t[:,1]]]
        EdgeDirectionTrig = numpy.sign(numpy.c_[t[:,1]-t[:,2],
                                                t[:,2]-t[:,0],
                                                t[:,0]-t[:,1]].astype(int))*(-1)
        mp_trig = 1/3*(p[t[:,0],:] + p[t[:,1],:] + p[t[:,2],:])
    else:
        edges_trigs = numpy.array([], dtype=numpy.int64).reshape(0,2)
        EdgeDirectionTrig= numpy.array([], dtype=numpy.int64).reshape(0,3)
        mp_trig = numpy.array([], dtype=numpy.int64).reshape(0,2)
    

    if q.size != 0:
        edges_quads = numpy.r_[numpy.c_[q[:,1],q[:,2]],
                               numpy.c_[q[:,2],q[:,3]],
                               numpy.c_[q[:,3],q[:,0]],
                               numpy.c_[q[:,0],q[:,1]]]
        EdgeDirectionQuad = numpy.sign(numpy.c_[q[:,1]-q[:,2],
                                                q[:,2]-q[:,3],
                                                q[:,3]-q[:,0],
                                                q[:,0]-q[:,1]].astype(int))*(-1)
        mp_quad = 1/4*(p[q[:,0],:] + p[q[:,1],:] + p[q[:,2],:] + p[q[:,3],:])
    else:
        edges_quads = numpy.array([], dtype=numpy.int64).reshape(0,2)
        EdgeDirectionQuad= numpy.array([], dtype=numpy.int64).reshape(0,4)
        mp_quad = numpy.array([], dtype=numpy.int64).reshape(0,2)

    e_new = numpy.sort(e[:,0:2])
    nt = t.shape[0]
    nq = q.shape[0]
    np = p.shape[0]
    ne = e_new.shape[0]

    
    #############################################################################################################
    edges = numpy.r_[numpy.sort(edges_trigs),numpy.sort(edges_quads)]
    EdgesToVertices, je = numpy.unique(edges,axis=0, return_inverse=True)

    NoEdges = EdgesToVertices.shape[0]
    TriangleToEdges = je[0:3*nt].reshape(nt,3, order='F')
    QuadToEdges = je[3*nt:].reshape(nq,4, order='F')
    #############################################################################################################
    

    #############################################################################################################
    # Need this so we can find the indices of the boundary edges in the global edge list
    BoundaryEdges2 = numpy.argwhere(numpy.bincount(je)==1)[:,0]
    _,je_new = numpy.unique(e_new, axis=0, return_inverse=True)
    BoundaryEdges = BoundaryEdges2[je_new]
    e_new = EdgesToVertices[BoundaryEdges,:]
    #############################################################################################################
    

    #############################################################################################################
    loc_trig,index_trig = ismember(TriangleToEdges,BoundaryEdges)
    loc_quad,index_quad = ismember(QuadToEdges,BoundaryEdges)

    indices_boundary = numpy.r_[index_trig,index_quad]
    direction_boundary = numpy.r_[EdgeDirectionTrig[loc_trig],EdgeDirectionQuad[loc_quad]]
    b = numpy.argsort(indices_boundary)
    #############################################################################################################

    
    #############################################################################################################
    MESH['Boundary'] = {}
    MESH['Boundary']['Region'] = e[:,2]
    MESH['Boundary']['p_index'] = numpy.unique(e)
    MESH['Boundary']['np'] = MESH['Boundary']['p_index'].size
    MESH['Boundary']['Edges'] = BoundaryEdges
    MESH['Boundary']['NoEdges'] = BoundaryEdges.shape[0]
    MESH['Boundary']['EdgeOrientation'] = direction_boundary[b]
    #############################################################################################################


    #############################################################################################################
    MESH['EdgesToVertices'] = EdgesToVertices
    MESH['TriangleToEdges'] = TriangleToEdges
    MESH['QuadToEdges'] = QuadToEdges
    MESH['NoEdges'] = NoEdges

    MESH['EdgeDirectionTrig'] = EdgeDirectionTrig
    MESH['EdgeDirectionQuad'] = EdgeDirectionQuad

    MESH['p'] = p
    MESH['e'] = e_new
    MESH['np'] = np
    MESH['ne'] = ne
    MESH['nq'] = nq

    MESH['mp'] = numpy.r_[mp_trig,mp_quad]
    #############################################################################################################
    
    Edges_Triangle_Mix = numpy.unique(TriangleToEdges)
    Edges_Quad_Mix = numpy.unique(QuadToEdges)
    MESH['Lists'] = {}
    MESH['Lists']['InterfaceTriangleQuad'] = numpy.intersect1d(Edges_Triangle_Mix,Edges_Quad_Mix)
    MESH['Lists']['JustTrig'] = numpy.setdiff1d(Edges_Triangle_Mix,MESH['Lists']['InterfaceTriangleQuad'])
    MESH['Lists']['JustQuad'] = numpy.setdiff1d(Edges_Quad_Mix,MESH['Lists']['InterfaceTriangleQuad'])

    loc,_ = ismember(QuadToEdges,MESH['Lists']['InterfaceTriangleQuad'])
    QuadsAtTriangleInterface = numpy.argwhere(loc)[:,0]
    QuadLayerEdges = numpy.unique(QuadToEdges[QuadsAtTriangleInterface,:])

    MESH['Lists']['QuadLayerEdges'] = QuadLayerEdges
    MESH['Lists']['QuadBoundaryEdges'] = numpy.intersect1d(MESH['Lists']['JustQuad'],MESH['Boundary']['Edges'])
    MESH['Lists']['QuadsAtTriangleInterface'] = QuadsAtTriangleInterface


    return MESH

def __pdemesh_trig(self,dpi=500,info=0):

    p = self.p
    t = self.t

    nt = self.t.shape[0]
    np = self.t.shape[0]

    x = np.array(([p[t[:,0],0],p[t[:,1],0],p[t[:,2],0],p[t[:,0],0]]))
    y = np.array(([p[t[:,0],1],p[t[:,1],1],p[t[:,2],1],p[t[:,0],1]]))

    plt.plot(x,y,'b-',linewidth=0.4)

    if info==1:
        for i in range(p.shape[0]):
            plt.text(p[i,0],p[i,1],np.r_[0:p.shape[0]][i], \
                        horizontalalignment='center',verticalalignment='center', \
                        backgroundcolor='yellow', \
                        bbox=dict(facecolor='yellow', edgecolor='none', alpha=1, pad=1), \
                        fontsize='x-small')        
        
        avg_x = 1/3*(x[1,:] + x[2,:] + x[3,:])
        avg_y = 1/3*(y[1,:] + y[2,:] + y[3,:])
        for i in range(t.shape[1]):
            plt.text(avg_x[i],avg_y[i],np.r_[0:t.shape[0]][i], \
                        horizontalalignment='center',verticalalignment='center', \
                        # backgroundcolor='yellow', \
                        # bbox=dict(facecolor='yellow', edgecolor='none', alpha=1, pad=1), \
                        fontsize='x-small')



def pdemesh(self,dpi=500,info=0):
    
    plt.rcParams['figure.dpi'] = 200 # Always do this first ... for some reason
    plt.rcParams['font.size'] = 8
    plt.gca().set_aspect('equal', adjustable='box')
    
    p = self.p
    t = self.t
    
    x = np.array([p[0,t[0,:]],p[0,t[1,:]],p[0,t[2,:]],p[0,t[0,:]]])
    y = np.array([p[1,t[0,:]],p[1,t[1,:]],p[1,t[2,:]],p[1,t[0,:]]])
    
    plt.plot(x,y,'b-',linewidth=0.4)
    
    if info==1:
        for i in range(p.shape[1]):
            plt.text(p[0,i],p[1,i],np.r_[0:p.shape[1]][i], \
                        horizontalalignment='center',verticalalignment='center', \
                        backgroundcolor='yellow', \
                        bbox=dict(facecolor='yellow', edgecolor='none', alpha=1, pad=1), \
                        fontsize='x-small')        
        
        avg_x = 1/3*(x[1,:] + x[2,:] + x[3,:])
        avg_y = 1/3*(y[1,:] + y[2,:] + y[3,:])
        for i in range(t.shape[1]):
            plt.text(avg_x[i],avg_y[i],np.r_[0:t.shape[1]][i], \
                        horizontalalignment='center',verticalalignment='center', \
                        # backgroundcolor='yellow', \
                        # bbox=dict(facecolor='yellow', edgecolor='none', alpha=1, pad=1), \
                        fontsize='x-small')
    
    if info==1:
        ued,_ = __make_edge_list(t)
        edge_mid = 1/2*(p[:,ued[0,:]] + p[:,ued[1,:]])
        for i in range(ued.shape[1]):
            plt.text(edge_mid[0,i],edge_mid[1,i],np.r_[0:ued.shape[1]][i], \
                        horizontalalignment='center',verticalalignment='center', \
                        backgroundcolor='gray', \
                        bbox=dict(facecolor='silver', edgecolor='none', alpha=1, pad=0.5), \
                        fontsize='x-small')   
        print('something')


def __make_edge_list(t):
    ed = numpy.array((numpy.r_[t[0,:],t[0,:],t[1,:]],
                      numpy.r_[t[1,:],t[2,:],t[2,:]]))

    sed = numpy.sort(ed,axis=0)

    ued, je = numpy.unique(sed.transpose(),axis=0, return_inverse=True)
    ued = ued.transpose().copy()

    ned = ed.shape[1]
    te = je.reshape((int(ned/3),3), order='F').transpose()
    
    return ued, te

def ismember(a_vec, b_vec):
    """ MATLAB equivalent ismember function """

    bool_ind = numpy.isin(a_vec,b_vec)
    common = a_vec[bool_ind]
    common_unique, common_inv  = numpy.unique(common, return_inverse=True)     # common = common_unique[common_inv]
    b_unique, b_ind = numpy.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
    common_ind = b_ind[numpy.isin(b_unique, common_unique, assume_unique=True)]
    return bool_ind, common_ind[common_inv]

# DEBUGGING
if __name__ == "__main__":
    from tools import *
    p,e,t,q = petq_from_gmsh(filename='mesh_new.geo',hmax=0.8)
    MESH = make_mesh_data_hybrid(p,e,t,q)
    MESH.pdemesh()
    print('dsa')