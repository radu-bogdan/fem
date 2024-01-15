
import numpy as npy # npy becauswe np is number of points... dont judge me :)
npy.set_printoptions(edgeitems=10, linewidth = 1000000)
# npy.set_printoptions(threshold = npy.inf)
# npy.set_printoptions(linewidth = npy.inf)
# npy.set_printoptions(precision=2)

# import pandas as pd
import plotly.io as pio
pio.renderers.default = "browser"

import plotly.graph_objects as go
# import plotly.colors as plyc
from scipy.interpolate import griddata
from . import lists as femlists
# import numba as jit
# from .tools import *
# from .tools import unique_rows

from .tools import getIndices

# import plotly.figure_factory as ff

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri

class mesh:
    # @profile       
    
    def __init__(self, p, e, t, q, r2d = npy.empty(0),r1d = npy.empty(0), identifications = npy.empty(0)):
        
        if t.size != 0:
            edges_trigs = npy.r_[npy.c_[t[:,1],t[:,2]],
                                 npy.c_[t[:,2],t[:,0]],
                                 npy.c_[t[:,0],t[:,1]]]
            EdgeDirectionTrig = npy.sign(npy.c_[t[:,1]-t[:,2],
                                                t[:,2]-t[:,0],
                                                t[:,0]-t[:,1]].astype(int))*(-1)
            mp_trig = 1/3*(p[t[:,0],:] + p[t[:,1],:] + p[t[:,2],:])
        else:
            edges_trigs = npy.empty(shape = (0,2), dtype = npy.int64)
            EdgeDirectionTrig = npy.empty(shape = (0,3), dtype = npy.int64)
            mp_trig = npy.empty(shape = (0,2), dtype = npy.int64)
        

        if q.size != 0:
            edges_quads = npy.r_[npy.c_[q[:,1],q[:,2]],
                                 npy.c_[q[:,2],q[:,3]],
                                 npy.c_[q[:,3],q[:,0]],
                                 npy.c_[q[:,0],q[:,1]]]
            EdgeDirectionQuad = npy.sign(npy.c_[q[:,1]-q[:,2],
                                                q[:,2]-q[:,3],
                                                q[:,3]-q[:,0],
                                                q[:,0]-q[:,1]].astype(int))*(-1)
            mp_quad = 1/4*(p[q[:,0],:] + p[q[:,1],:] + p[q[:,2],:] + p[q[:,3],:])
        else:
            edges_quads = npy.empty(shape = (0,2), dtype = npy.int64)
            EdgeDirectionQuad = npy.empty(shape = (0,4), dtype = npy.int64)
            mp_quad = npy.empty(shape = (0,2), dtype = npy.int64)
            
        # e_new = e[:,0:2].copy()
        e_new = npy.sort(e[:,:2])
            
        nt = t.shape[0]
        nq = q.shape[0]
        
        #############################################################################################################
        edges = npy.r_[npy.sort(edges_trigs),
                       npy.sort(edges_quads)].astype(int)
        EdgesToVertices, je = npy.unique(edges, axis = 0, return_inverse = True)
        # EdgesToVertices, je = unique_rows(edges, return_inverse = True)
        
        NoEdges = EdgesToVertices.shape[0]
        TriangleToEdges = je[0:3*nt].reshape(nt,3, order = 'F').astype(npy.int64)
        QuadToEdges = je[3*nt:].reshape(nq,4, order = 'F')
        BoundaryEdges = intersect2d(EdgesToVertices,e_new)
        # InteriorEdges = npy.setdiff1d(npy.arange(NoEdges),BoundaryEdges)
        
        EdgesToVertices = npy.c_[EdgesToVertices,npy.zeros(EdgesToVertices.shape[0],dtype = npy.int64)-1]
        EdgesToVertices[BoundaryEdges,2] = e[:,-1]
        #############################################################################################################
        
        
        
        #############################################################################################################
        tte_flat = npy.argsort(TriangleToEdges.flatten())
        tte_sort = npy.sort(TriangleToEdges.flatten())
        _,c = npy.unique(tte_sort, return_counts = True)
        indices_single = npy.argwhere(c==1)[:,0]
        
        all_indices_pos = npy.arange(tte_flat.size)
        indices_single_pos = 2*indices_single-npy.arange(indices_single.size)
        
        indices_double_pos = npy.setdiff1d(all_indices_pos, indices_single_pos)
        
        tte_sort_ind = tte_sort[indices_double_pos]
        tte_flat_ind = tte_flat[indices_double_pos]
        
        IntEdgesToTriangles = npy.c_[npy.reshape(tte_flat_ind,(tte_flat_ind.size//2,2))//3, npy.unique(tte_sort_ind)]
        #############################################################################################################
        
        if t.shape[1]==7:
            maxp = t[:,:3].max()
            pm = p.copy()
            p = p[:maxp+1,:]
        else:
            pm = p.copy()
            
        #############################################################################################################

        self.EdgesToVertices = EdgesToVertices
        self.TriangleToEdges = TriangleToEdges
        self.QuadToEdges = QuadToEdges
        self.NoEdges = NoEdges
        self.EdgeDirectionTrig = EdgeDirectionTrig
        self.EdgeDirectionQuad = EdgeDirectionQuad
        self.Boundary_Region = e[:,-1]
        self.Boundary_Edges = BoundaryEdges
        self.Boundary_NoEdges = BoundaryEdges.shape[0]
        self.IntEdgesToTriangles = IntEdgesToTriangles[:,0:2]
        self.NonSingle_Edges = IntEdgesToTriangles[:,2]

        self.p = p; self.np = p.shape[0]
        self.e = npy.c_[e_new,e[:,-1]]; self.ne = e_new.shape[0]
        self.t = t[:,npy.r_[:3,-1]]; self.nt = nt
        self.q = q; self.nq = nq
        self.mp = npy.r_[mp_trig,mp_quad]
        self.regions_2d = list(r2d)
        self.regions_1d = list(r1d)
        self.identifications = identifications
        
        self.FEMLISTS = {}
        #############################################################################################################
        
        if t.shape[1]==4:
            t0 = self.t[:,0]; t1 = self.t[:,1]; t2 = self.t[:,2]
            C00 = pm[t0,0]; C01 = pm[t1,0]; C02 = pm[t2,0];
            C10 = pm[t0,1]; C11 = pm[t1,1]; C12 = pm[t2,1];
            
            self.Fx = lambda x,y : C00*(1-x-y) + C01*x + C02*y
            self.Fy = lambda x,y : C10*(1-x-y) + C11*x + C12*y
            
            self.JF00 = lambda x,y : C01-C00
            self.JF01 = lambda x,y : C02-C00
            self.JF10 = lambda x,y : C11-C10
            self.JF11 = lambda x,y : C12-C10
        
        if t.shape[1]==7:
            t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]; t3 = t[:,3]; t4 = t[:,4]; t5 = t[:,5]
            C00 = pm[t0,0]; C01 = pm[t1,0]; C02 = pm[t2,0]; C03 = pm[t3,0]; C04 = pm[t4,0]; C05 = pm[t5,0];
            C10 = pm[t0,1]; C11 = pm[t1,1]; C12 = pm[t2,1]; C13 = pm[t3,1]; C14 = pm[t4,1]; C15 = pm[t5,1];
            
            self.Fx = lambda x,y : C00*(1-x-y)*(1-2*x-2*y) + C01*x*(2*x-1) + C02*y*(2*y-1) + C03*4*x*y + C04*4*y*(1-x-y) + C05*4*x*(1-x-y)
            self.Fy = lambda x,y : C10*(1-x-y)*(1-2*x-2*y) + C11*x*(2*x-1) + C12*y*(2*y-1) + C13*4*x*y + C14*4*y*(1-x-y) + C15*4*x*(1-x-y)
            
            self.JF00 = lambda x,y : C00*(4*x+4*y-3) + C01*(4*x-1) + C02*(0*x)   + C03*(4*y) + C04*(-4*y)         + C05*(-4*(2*x+y-1))
            self.JF01 = lambda x,y : C00*(4*x+4*y-3) + C01*(0*x)   + C02*(4*y-1) + C03*(4*x) + C04*(-4*(x+2*y-1)) + C05*(-4*x)
            self.JF10 = lambda x,y : C10*(4*x+4*y-3) + C11*(4*x-1) + C12*(0*x)   + C13*(4*y) + C14*(-4*y)         + C15*(-4*(2*x+y-1))
            self.JF11 = lambda x,y : C10*(4*x+4*y-3) + C11*(0*x)   + C12*(4*y-1) + C13*(4*x) + C14*(-4*(x+2*y-1)) + C15*(-4*x)
        
            
        self.detA = lambda x,y : self.JF00(x,y)*self.JF11(x,y)-self.JF01(x,y)*self.JF10(x,y)
        
        self.iJF00 = lambda x,y:  1/self.detA(x,y)*self.JF11(x,y)
        self.iJF01 = lambda x,y: -1/self.detA(x,y)*self.JF01(x,y)
        self.iJF10 = lambda x,y: -1/self.detA(x,y)*self.JF10(x,y)
        self.iJF11 = lambda x,y:  1/self.detA(x,y)*self.JF00(x,y)
        
        #############################################################################################################
        
        t0 = self.t[:,0]; t1 = self.t[:,1]; t2 = self.t[:,2]
        A00 = self.p[t1,0]-self.p[t0,0]; A01 = self.p[t2,0]-self.p[t0,0]
        A10 = self.p[t1,1]-self.p[t0,1]; A11 = self.p[t2,1]-self.p[t0,1]
        # self.detA = A00*A11-A01*A10
        
        
        nor1 = npy.r_[1,1]; nor2 = npy.r_[-1,0]; nor3 = npy.r_[0,-1]
        
        # normal times the size of the edge
        normal_e0_0 = (A11*nor1[0]-A10*nor1[1]); normal_e0_1 = -A01*nor1[0]+A00*nor1[1]
        normal_e1_0 = (A11*nor2[0]-A10*nor2[1]); normal_e1_1 = -A01*nor2[0]+A00*nor2[1]
        normal_e2_0 = (A11*nor3[0]-A10*nor3[1]); normal_e2_1 = -A01*nor3[0]+A00*nor3[1]
        
        tan1 = npy.r_[1,-1]; tan2 = npy.r_[0,1]; tan3 = npy.r_[-1,0]
        
        # tangent times the size of the edge
        tangent_e0_0 = A00*tan1[0]+A01*tan1[1]; tangent_e0_1 = A10*tan1[0]+A11*tan1[1] 
        tangent_e1_0 = A00*tan2[0]+A01*tan2[1]; tangent_e1_1 = A10*tan2[0]+A11*tan2[1]
        tangent_e2_0 = A00*tan3[0]+A01*tan3[1]; tangent_e2_1 = A10*tan3[0]+A11*tan3[1]
        
        # normalization
        len_e0 = npy.sqrt(tangent_e0_0**2+tangent_e0_1**2)
        len_e1 = npy.sqrt(tangent_e1_0**2+tangent_e1_1**2)
        len_e2 = npy.sqrt(tangent_e2_0**2+tangent_e2_1**2)
        
        self.A00 = A00
        self.A10 = A10
        self.A01 = A01
        self.A11 = A11
        
        self.tangent0 = npy.c_[tangent_e0_0/len_e0,tangent_e1_0/len_e1,tangent_e2_0/len_e2]
        self.tangent1 = npy.c_[tangent_e0_1/len_e0,tangent_e1_1/len_e1,tangent_e2_1/len_e2]
        
        self.normal0 = npy.c_[normal_e0_0/len_e0,normal_e1_0/len_e1,normal_e2_0/len_e2]
        self.normal1 = npy.c_[normal_e0_1/len_e0,normal_e1_1/len_e1,normal_e2_1/len_e2]
        
        self.len_e = npy.c_[len_e0,len_e1,len_e2]
        
        
        #############################################################################################################
        
    def __repr__(self):
        return f"np:{self.np}, nt:{self.nt}, nq:{self.nq}, ne:{self.ne}, ne_all:{self.NoEdges}"
    
    def from_netgen(self,geoOCCmesh):
        npoints = geoOCCmesh.Elements2D().NumPy()['np'].max()

        p = geoOCCmesh.Coordinates()
        t = npy.c_[geoOCCmesh.Elements2D().NumPy()['nodes'].astype(npy.uint64)[:,:npoints],
                   geoOCCmesh.Elements2D().NumPy()['index'].astype(npy.uint64)]-1

        e = npy.c_[geoOCCmesh.Elements1D().NumPy()['nodes'].astype(npy.uint64)[:,:((npoints+1)//2)],
                   geoOCCmesh.Elements1D().NumPy()['index'].astype(npy.uint64)]-1

        max_bc_index = geoOCCmesh.Elements1D().NumPy()['index'].astype(npy.uint64).max()
        max_rg_index = geoOCCmesh.Elements2D().NumPy()['index'].astype(npy.uint64).max()

        q = npy.empty(0)
        
        regions_1d_np = []
        for i in range(max_bc_index):
            regions_1d_np += [geoOCCmesh.GetBCName(i)]

        regions_2d_np = []
        for i in range(max_rg_index):
            regions_2d_np += [geoOCCmesh.GetMaterial(i+1)]
        
        identifications = npy.array(geoOCCmesh.GetIdentifications())
        
        return p, e, t, q, regions_2d_np, regions_1d_np, identifications
    
    @classmethod
    def netgen(cls, geoOCCmesh):
        # cls.geoOCCmesh = geoOCCmesh
        p, e, t, q, regions_2d_np, regions_1d_np, identifications = cls.from_netgen(cls,geoOCCmesh)
        return cls(p, e, t, q, regions_2d_np, regions_1d_np, identifications)
        # self.__init__(self, p,e,t,q,r2d = regions_2d_np,r1d = regions_1d_np)
    
    # @profile
    def makeBEO(self): # Boundary Edge Orientations
        gem_list = npy.r_[self.TriangleToEdges.ravel(),self.QuadToEdges.ravel()]
        gem_dir_list = npy.r_[self.EdgeDirectionTrig.ravel(),self.EdgeDirectionQuad.ravel()]
        
        _,j,i = npy.intersect1d(gem_list,self.Boundary_Edges, return_indices = True)
        
        Boundary_EdgeOrientation = npy.zeros(shape = i.shape)
        Boundary_EdgeOrientation[i] = gem_dir_list[j]
        self.Boundary_EdgeOrientation = Boundary_EdgeOrientation
        
    
    def makeRest(self):
        #############################################################################################################
        # loc_trig,index_trig = self.__ismember(self.TriangleToEdges,self.Boundary_Edges)
        # loc_quad,index_quad = self.__ismember(self.QuadToEdges,self.Boundary_Edges)

        # indices_boundary = npy.r_[index_trig,index_quad]
        # direction_boundary = npy.r_[self.EdgeDirectionTrig[loc_trig],self.EdgeDirectionQuad[loc_quad]]
        # b = npy.argsort(indices_boundary)
        
        # self.Boundary_EdgeOrientation = direction_boundary[b]
        
        #############################################################################################################
        
        Edges_Triangle_Mix = npy.unique(self.TriangleToEdges)
        Edges_Quad_Mix = npy.unique(self.QuadToEdges)
        self.Lists_InterfaceTriangleQuad = npy.intersect1d(Edges_Triangle_Mix,Edges_Quad_Mix)
        self.Lists_JustTrig = npy.setdiff1d(Edges_Triangle_Mix,self.Lists_InterfaceTriangleQuad)
        self.Lists_JustQuad = npy.setdiff1d(Edges_Quad_Mix,self.Lists_InterfaceTriangleQuad)
        self.Lists_TrigBoundaryEdges = npy.intersect1d(self.Lists_JustTrig,self.Boundary_Edges)
        
        loc,_ = self.__ismember(self.QuadToEdges,self.Lists_InterfaceTriangleQuad)
        QuadsAtTriangleInterface = npy.argwhere(loc)[:,0]
        QuadLayerEdges = npy.unique(self.QuadToEdges[QuadsAtTriangleInterface,:])
    
        self.Lists_QuadLayerEdges = QuadLayerEdges
        self.Lists_QuadBoundaryEdges = npy.intersect1d(self.Lists_JustQuad,self.Boundary_Edges)
        self.Lists_QuadsAtTriangleInterface = QuadsAtTriangleInterface
        
        #############################################################################################################
        
        regions_to_points = npy.empty(shape = [0,2], dtype = 'int64')
        if self.t.shape[1]>3:
            for i in range(max(self.t[:,3])):
                indices = npy.unique(self.t[npy.argwhere(self.t[:,3] == (i+1))[:,0],:])            
                vtr = npy.c_[(i+1)*npy.ones(shape = indices.shape, dtype = 'int64'), indices]
                regions_to_points = npy.r_[regions_to_points, vtr]
            
        self.RegionsToPoints = regions_to_points
            
    def makeFemLists(self,space):
        self.FEMLISTS = femlists.lists(self, space)
        
    def refinemesh(self,classic = True):
        
        if (hasattr(self, "geoOCCmesh") and classic == False):
            print("geoOCC")
            self.geoOCCmesh.Refine()
            p_new, e_new, t_new, q_new, regions_2d_np, regions_1d_np = self.from_netgen(self.geoOCCmesh)
            self.__init__(p_new,e_new,t_new,q_new,regions_2d_np,regions_1d_np)
            self.FEMLISTS = {} # reset fem lists, cuz new mesh
            # self.delattr("Boundary_EdgeOrientation")
            
            if hasattr(self, "Boundary_EdgeOrientation"):
                del self.Boundary_EdgeOrientation
        else:
            pn = 1/2*(self.p[self.EdgesToVertices[:,0],:]+
                      self.p[self.EdgesToVertices[:,1],:])
            p_new = npy.r_[self.p,pn]
            
            tn = self.np + self.TriangleToEdges
            t_new = npy.r_[npy.c_[self.t[:,0],tn[:,2],tn[:,1],self.t[:,3]],
                           npy.c_[self.t[:,1],tn[:,0],tn[:,2],self.t[:,3]],
                           npy.c_[self.t[:,2],tn[:,1],tn[:,0],self.t[:,3]],
                           npy.c_[    tn[:,0],tn[:,1],tn[:,2],self.t[:,3]]].astype(npy.int64)
            bn = self.np + self.Boundary_Edges
            e_new = npy.r_[npy.c_[self.e[:,0],bn,self.Boundary_Region],
                           npy.c_[bn,self.e[:,1],self.Boundary_Region]].astype(npy.int64)
            q_new = self.q
            
            self.__init__(p_new,e_new,t_new,q_new,self.regions_2d,self.regions_1d)
            self.FEMLISTS = {} # reset fem lists, cuz new mesh
            # self.delattr("Boundary_EdgeOrientation")
            
            if hasattr(self, "Boundary_EdgeOrientation"):
                del self.Boundary_EdgeOrientation

        
        print('Generated refined mesh with ' + str(p_new.shape[0]) + ' points, ' 
                                             + str(e_new.shape[0]) + ' boundary edges, ' 
                                             + str(t_new.shape[0]) + ' triangles, ' 
                                             + str(q_new.shape[0]) + ' quadrilaterals.')
    
    def rho(self):
        x = self.p[self.t[:,1],:]-self.p[self.t[:,2],:]
        y = self.p[self.t[:,2],:]-self.p[self.t[:,0],:]
        z = self.p[self.t[:,0],:]-self.p[self.t[:,1],:]
        
        len_x = npy.sqrt(x[:,0]**2 + x[:,1]**2)
        len_y = npy.sqrt(y[:,0]**2 + y[:,1]**2)
        len_z = npy.sqrt(z[:,0]**2 + z[:,1]**2)
        
        return npy.sqrt(len_x*len_y*len_z/(len_x+len_y+len_z))
        # return len_x
    
    def h(self):
        x = self.p[self.t[:,1],:]-self.p[self.t[:,2],:]
        y = self.p[self.t[:,2],:]-self.p[self.t[:,0],:]
        z = self.p[self.t[:,0],:]-self.p[self.t[:,1],:]
        
        len_x = npy.sqrt(x[:,0]**2 + x[:,1]**2)
        len_y = npy.sqrt(y[:,0]**2 + y[:,1]**2)
        len_z = npy.sqrt(z[:,0]**2 + z[:,1]**2)
        
        len_mat = npy.c_[len_x,len_y,len_z]
        
        return npy.max(len_mat, axis=1)
        # return len_mat
    
    def refine(self,f):
        # if f.size != 3*self.nt:
        #     print('Wrong size! must be P1-discontinuous')
        #     return        
        return npy.r_[f,f,f,f]
    
    def trianglesFromPoints(self, points):
        return npy.unique(npy.where(npy.in1d(self.t[:,0:3].ravel(),points))[0]//3)
        
    def __ismember(self,a_vec, b_vec):
        """ MATLAB equivalent ismember function """
        bool_ind = npy.isin(a_vec,b_vec)
        common = a_vec[bool_ind]
        common_unique, common_inv  = npy.unique(common, return_inverse=True)     # common = common_unique[common_inv]
        b_unique, b_ind = npy.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
        common_ind = b_ind[npy.isin(b_unique, common_unique, assume_unique=True)]
        return bool_ind, common_ind[common_inv]
        
    def pdegeom(self,**kwargs):
        if "ax" not in kwargs:
            # create new figure
            fig = plt.figure(**{k: v for k, v in kwargs.items() if k in ['figsize']})
            ax = fig.add_subplot(111)
            aspect = kwargs["aspect"] if "aspect" in kwargs else 1.0
            ax.set_aspect(aspect)
            # ax.set_axis_off()
        else:
            ax = kwargs["ax"]
            # ax.clear()
        
        xs = []; ys = []
        for ss, tt, uu, vv in zip(self.p[self.e[:,0],0],
                                  self.p[self.e[:,0],1],
                                  self.p[self.e[:,1],0],
                                  self.p[self.e[:,1],1]):
            xs.append(ss); xs.append(uu); xs.append(None)
            ys.append(tt); ys.append(vv); ys.append(None)
        
        ax.plot(xs,ys, linewidth = 0.7, color = 'red')
        return ax
    
    def pdesurf2(self,fun,**kwargs):
        
        vmin = npy.min(fun)
        vmax = npy.max(fun)
        
        if 'cmin' in kwargs.keys(): vmin = kwargs['cmin']
        if 'cmax' in kwargs.keys(): vmax = kwargs['cmax']
        
        if "ax" not in kwargs:
            # create new figure
            fig = plt.figure(**{k: v for k, v in kwargs.items() if k in ['figsize']})
            ax = fig.add_subplot(111)
            aspect = kwargs["aspect"] if "aspect" in kwargs else 1.0
            ax.set_aspect(aspect)
            # ax.set_axis_off()
        else: ax = kwargs["ax"]
        
        
        if "cmap" not in kwargs: cmap = plt.cm.jet
        else: cmap = kwargs["cmap"]
        
        
        if (fun.size==self.np):
            Triang = matplotlib.tri.Triangulation(self.p[:,0], self.p[:,1], self.t[:,0:3])
            chip = ax.tripcolor(Triang, fun, cmap = cmap, lw = 0.1, shading='gouraud', vmin = vmin, vmax = vmax)
            
            # refiner = tri.UniformTriRefiner(Triang)
            # tri_refi, z_test_refi = refiner.refine_field(fun, subdiv=3)
            # chip = ax.tricontour(tri_refi, z_test_refi, colors = 'k')
            # chip = ax.tricontour(Triang, fun, colors='k')
            
        if (fun.size==self.nt):
            Triang = matplotlib.tri.Triangulation(self.p[:,0], self.p[:,1], self.t[:,0:3])
            chip = ax.tripcolor(Triang, fun, cmap = cmap, lw = 0.1, vmin = vmin, vmax = vmax)
            
        if (fun.size==3*self.nt):
            p0 = self.p[:,0]; p1 = self.p[:,1]
            t = self.t[:,0:3]; nt = self.nt
            
            p0d = npy.c_[p0[t[:,0]],p0[t[:,1]],p0[t[:,2]]].ravel()
            p1d = npy.c_[p1[t[:,0]],p1[t[:,1]],p1[t[:,2]]].ravel()
            td = npy.r_[:3*nt].reshape(nt,3)
            Triang = matplotlib.tri.Triangulation(p0d, p1d, td)
            chip = ax.tripcolor(Triang, fun, cmap = cmap, lw = 0.1, shading='gouraud', vmin = vmin, vmax = vmax)
        
        if "cbar" in kwargs.keys() and kwargs["cbar"]==1:
            plt.colorbar(chip)
        return ax
    
    def pdemesh2(self,**kwargs):
        if "ax" not in kwargs:
            # create new figure
            fig = plt.figure(**{k: v for k, v in kwargs.items() if k in ['figsize']})
            ax = fig.add_subplot(111)
            aspect = kwargs["aspect"] if "aspect" in kwargs else 1.0
            ax.set_aspect(aspect)
            # ax.set_axis_off()
        else:
            ax = kwargs["ax"]
            # ax.clear()
        
        xx_trig = npy.c_[self.p[self.t[:,0],0], self.p[self.t[:,1],0], self.p[self.t[:,2],0]]
        yy_trig = npy.c_[self.p[self.t[:,0],1], self.p[self.t[:,1],1], self.p[self.t[:,2],1]]
        
        xxx_trig = npy.c_[xx_trig, xx_trig[:,0], npy.nan*xx_trig[:,0]]
        yyy_trig = npy.c_[yy_trig, yy_trig[:,0], npy.nan*yy_trig[:,0]]
        
        ax.plot(
            xxx_trig.ravel(), 
            yyy_trig.ravel(),
            color = 'k',
            linewidth = 0.1
            # kwargs['color'] if 'color' in kwargs else 'k',
            # linewidth = kwargs['linewidth'] if 'linewidth' in kwargs else .5,
        )
        # ax.set_axis_off()
        # ax.show = lambda: plt.show()
        return ax
    
    def pdemesh(self,dpi=500,info=0,border=0):

        p = self.p; t = self.t; q = self.q

        fig = go.Figure()

        if t.size != 0:
            fig = self.__pdemesh_trig(fig,info)

        if q.size != 0:
            fig = self.__pdemesh_quad(fig,info)

        if info == 1:
            fig.add_trace(go.Scattergl(mode='text+markers',
                                    name='Points',
                                    x = p[:,0], 
                                    y = p[:,1], 
                                    text = list(range(0, len(p))),
                                    marker=dict(
                                        color='moccasin',
                                        size=25,
                                        line=dict(
                                            color='black',
                                            width=1
                                        )
                                        ),
                                    textfont=dict(color='#000000',size=13), 
                                    fill = "none"))
            fig.add_trace(go.Scattergl(mode='text+markers',
                                    name='Edges',
                                    x = 1/2*(p[self.EdgesToVertices[:,0],0]+p[self.EdgesToVertices[:,1],0]),
                                    y = 1/2*(p[self.EdgesToVertices[:,0],1]+p[self.EdgesToVertices[:,1],1]),
                                    text = list(range(0, self.NoEdges)),
                                    marker=dict(
                                        color='cyan',
                                        size=25,
                                        line=dict(
                                            color='black',
                                            width=1
                                        )
                                        ),
                                    textfont=dict(color='#000000',size=13), 
                                    fill = "none"))
            
    
        fig.update_xaxes(scaleanchor = "y", scaleratio = 1)
        fig.update_xaxes(range=[npy.min(p[:,0])-border,npy.max(p[:,0])+border])
        fig.update_yaxes(range=[npy.min(p[:,1])-border,npy.max(p[:,1])+border])

        fig.update_layout(plot_bgcolor = "white",
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False))
        # fig.show()
        return fig

    def __pdemesh_trig(self,fig,info):
        
        p = self.p; t = self.t[:,0:3]
        
        # x = npy.c_[p[t[:,0],0],p[t[:,1],0],p[t[:,2],0],p[t[:,0],0],npy.nan*p[t[:,0],0]].flatten()#.tolist()
        # y = npy.c_[p[t[:,0],1],p[t[:,1],1],p[t[:,2],1],p[t[:,0],1],npy.nan*p[t[:,0],1]].flatten()#.tolist()
                        
        xx_trig = npy.c_[p[t[:,0],0],p[t[:,1],0],p[t[:,2],0]]
        yy_trig = npy.c_[p[t[:,0],1],p[t[:,1],1],p[t[:,2],1]]
        
        xxx_trig = npy.c_[xx_trig,xx_trig[:,0],npy.nan*xx_trig[:,0]]
        yyy_trig = npy.c_[yy_trig,yy_trig[:,0],npy.nan*yy_trig[:,0]]
        
        # ind = npy.argsort(x)
        # print(ind.shape,ind.dtype)
        # print(ind.shape)
        
        fig.add_trace(go.Scatter(mode='lines', 
                                   name='TrigTraces',
                                   # x=x,
                                   # y=y,
                                   x=xxx_trig.ravel(),
                                   y=yyy_trig.ravel(),
                                   line=dict(color='blue', 
                                             width=2)))
        if info == 1:
            fig.add_trace(go.Scattergl(mode='text+markers',
                                       name='Triangles',
                                       x = 1/3*(p[t[:,0],0]+p[t[:,1],0]+p[t[:,2],0]),
                                       y = 1/3*(p[t[:,0],1]+p[t[:,1],1]+p[t[:,2],1]),
                                       text = list(range(0, t.shape[0])),
                                       marker = dict(
                                           color = 'yellow',
                                           size = 25,
                                           line = dict(
                                               color = 'black',
                                               width = 1
                                               )
                                           ),
                                    textfont = dict(color = '#000000',size = 13), 
                                    fill = "none"))
        return fig
        
    def __pdemesh_quad(self,fig,info):
        
        p = self.p; q = self.q
        
        x = npy.c_[p[q[:,0],0],p[q[:,1],0],p[q[:,2],0],p[q[:,3],0],p[q[:,0],0],npy.nan*p[q[:,0],0]].ravel().tolist()
        y = npy.c_[p[q[:,0],1],p[q[:,1],1],p[q[:,2],1],p[q[:,3],1],p[q[:,0],1],npy.nan*p[q[:,0],1]].ravel().tolist()                
        
        fig.add_trace(go.Scattergl(mode='lines', 
                                 name='QuadTraces',
                                 x=x, 
                                 y=y, 
                                 line=dict(color='blue', 
                                           width=2)))
        
        if info == 1:
            fig.add_trace(go.Scattergl(mode='text+markers',
                                     name='Quads',
                                     x = 1/4*(p[q[:,0],0]+p[q[:,1],0]+p[q[:,2],0]+p[q[:,3],0]),
                                     y = 1/4*(p[q[:,0],1]+p[q[:,1],1]+p[q[:,2],1]+p[q[:,3],1]),
                                     text = list(range(self.nt, self.nq+self.nt)),
                                     marker=dict(
                                         color='yellow',
                                         size=25,
                                         line=dict(
                                             color='black',
                                             width=1
                                         )
                                         ),
                                     textfont=dict(color='#000000',size=13), 
                                     fill = "none"))
        return fig
    
    def pdesurf(self,u,**kwargs):
        
        cmin = npy.min(u)
        cmax = npy.max(u)
        
        if 'cmin' in kwargs.keys():
            cmin = kwargs['cmin']
        if 'cmax' in kwargs.keys():
            cmax = kwargs['cmax']
        
        nt = self.nt; p = self.p; t = self.t[:,0:3]; np = self.np
        
        xx_trig = npy.c_[p[t[:,0],0],p[t[:,1],0],p[t[:,2],0]]
        yy_trig = npy.c_[p[t[:,0],1],p[t[:,1],1],p[t[:,2],1]]
        
        if u.size == 3*nt:
            zz = u[:3*nt].reshape(nt,3)
        if u.size == nt:
            zz = npy.tile(u[:nt],(3,1)).T
        if u.size == np:
            zz = u[t]
        
        fig = go.Figure()
        
        ii, jj, kk = npy.r_[:3*nt].reshape((nt, 3)).T
        fig.add_trace(go.Mesh3d(
            name = 'Trig values',
            x = xx_trig.ravel(), y = yy_trig.ravel(), z = 0*zz.ravel(),
            i = ii, j = jj, k = kk, intensity = zz.ravel(), 
            colorscale = 'Jet',
            cmin = cmin,
            cmax = cmax,
            intensitymode = 'vertex',
            lighting = dict(ambient = 1),
            contour_width = 1, contour_color = "#000000", contour_show = True,
        ))
        
        xxx_trig = npy.c_[xx_trig,xx_trig[:,0],npy.nan*xx_trig[:,0]]
        yyy_trig = npy.c_[yy_trig,yy_trig[:,0],npy.nan*yy_trig[:,0]]
        zzz_trig = npy.c_[zz,zz[:,0],npy.nan*zz[:,0]]
        
        fig.add_trace(go.Scatter3d(name = 'Trig traces',
                                    mode = 'lines',
                                    x = xxx_trig.ravel(),
                                    y = yyy_trig.ravel(),
                                    z = 0*zzz_trig.ravel(),
                                    line = go.scatter3d.Line(color = 'black', 
                                                            width = 1.5),
                                    showlegend = False))
        
        camera = dict(eye = dict(x = 0, y = -0.0001, z = 1))
        ratio = (max(self.p[:,0])-min(self.p[:,0]))/(max(self.p[:,1])-min(self.p[:,1]))
        scene = dict(aspectratio = dict(x = ratio, y = 1, z = 1),
                           xaxis = dict(showspikes = False, visible=False),
                           yaxis = dict(showspikes = False, visible=False),
                           zaxis = dict(showspikes = False, visible=False))
        
        fig.update_layout(
                          scene = scene,
                          scene_camera = camera,
                          margin_l=0,
                          margin_t=0,
                          margin_r=0,
                          margin_b=0,
                          legend = dict(yanchor = "top",
                                          y = 1,
                                          xanchor = "left",
                                          x = 0,
                                          bgcolor = "LightSteelBlue",
                                          bordercolor = "Black",
                                          borderwidth = 2),
                          )
        fig.update_traces(showlegend = True)
        color_scales_3D = ['Blackbody','Bluered','Blues','Cividis','Earth','Electric','Greens','Greys','Hot','Jet','Picnic','Portland','Rainbow','RdBu','Reds','Viridis','YlGnBu','YlOrRd','Brwnyl']
        list_color_scales = list(map(lambda x: dict(args=["colorscale", x],label=x,method="restyle"),color_scales_3D))
                    
        fig.update_layout(
        updatemenus = [
            dict(
                buttons = list_color_scales,
                direction = "down",
                showactive = False,
                xanchor = "left",
                yanchor = "top"),
            dict(
                buttons = list([
                    dict(
                        args = ["scene.camera.projection.type", "orthographic"],
                        label = "Orthographic",
                        method = "relayout"
                    ),
                    dict(
                        args = ["scene.camera.projection.type", "perspective"],
                        label = "Perspective",
                        method = "relayout"
                    ),
                ]),
                direction="down",
                showactive=True,
                xanchor="left",
                yanchor="bottom")],
                      margin_l=0,
                      margin_t=0,
                      margin_r=0,
                      margin_b=0,)
        
        
        return fig
    
    def pdesurf_hybrid(self,TrigQuadDict,u,u_height=1):
        
        DATAT = TrigQuadDict.get('trig')
        DATAQ = TrigQuadDict.get('quad')
        controls = TrigQuadDict.get('controls')
        
        fig = go.Figure()
        
        if self.t.shape[0]!=0:
            fig = self.__pdesurf_trig(fig,DATAT,u,u_height)
        
        if self.q.shape[0]!=0:
            fig = self.__pdesurf_quad(fig,DATAQ,u,u_height)
        
        
        camera = dict(eye = dict(x = 0, y = -0.0001, z = 1))
        ratio = (max(self.p[:,0])-min(self.p[:,0]))/(max(self.p[:,1])-min(self.p[:,1]))
        scene = dict(aspectratio = dict(x = ratio, y = 1, z = 1),
                           xaxis = dict(showspikes = False, visible=False),
                           yaxis = dict(showspikes = False, visible=False),
                           zaxis = dict(showspikes = False, visible=False))
        # camera = dict(eye = dict(x = -1e-10, y = -1e-10, z = 1e10))
        fig.update_layout(
                          scene = scene,
                          scene_camera = camera,
                          # hovermode = "x",
                          # autosize = True,
                          # dragmode = 'select',
                          margin_l=0,
                          margin_t=0,
                          margin_r=0,
                          margin_b=0,
                          legend = dict(yanchor = "top",
                                          y = 1,
                                          xanchor = "left",
                                          x = 0,
                                          bgcolor = "LightSteelBlue",
                                          bordercolor = "Black",
                                          borderwidth = 2),
                          )
        # fig.layout.scene.camera.projection.type = "orthographic"
        # fig.layout.paper_bgcolor = "#7f7f7f"
        # fig.layout.plot_bgcolor = "#c7c7c7"
        fig.update_traces(showlegend = True)
        
        scene = dict(aspectratio = dict(x = ratio, y = 1, z = 1),
                           xaxis = dict(showspikes = False, visible=False),
                           yaxis = dict(showspikes = False, visible=False),
                           zaxis = dict(showspikes = False, visible=False))
        
        # border = 0
        # fig.update_xaxes(scaleanchor = "y", scaleratio = 1)
        # fig.update_xaxes(range=[npy.min(self.p[:,0])-border,npy.max(self.p[:,0])+border])
        # fig.update_yaxes(range=[npy.min(self.p[:,1])-border,npy.max(self.p[:,1])+border])
        
        if controls == 1:
            
            # color_scales = plyc.named_colorscales()
            color_scales_3D = ['Blackbody','Bluered','Blues','Cividis','Earth','Electric','Greens','Greys','Hot','Jet','Picnic','Portland','Rainbow','RdBu','Reds','Viridis','YlGnBu','YlOrRd']
            list_color_scales = list(map(lambda x: dict(args=["colorscale", x],label=x,method="restyle"),color_scales_3D))
                        
            fig.update_layout(
            updatemenus = [
                dict(
                    buttons = list_color_scales,
                    # type = "buttons",
                    direction = "down",
                    # pad={"r": 10, "t": 10},
                    showactive = False,
                    # x = 0,
                    xanchor = "left",
                    # y = 0,
                    yanchor = "top"),
                dict(
                    buttons = list([
                        dict(
                            args = ["scene.camera.projection.type", "orthographic"],
                            label = "Orthographic",
                            method = "relayout"
                        ),
                        dict(
                            args = ["scene.camera.projection.type", "perspective"],
                            label = "Perspective",
                            method = "relayout"
                        ),
                    ]),
                    # type = "buttons",
                    direction="down",
                    # pad={"r": 10, "t": 10},
                    showactive=True,
                    # x=0.2,
                    xanchor="left",
                    # y=0,
                    yanchor="bottom")],
                          margin_l=0,
                          margin_t=0,
                          margin_r=0,
                          margin_b=0,)
        
        
        return fig
            

    def __pdesurf_trig(self,fig,DATAT,u,u_height):
        nt = self.nt; p = self.p; t = self.t[:,0:3];
        
        xx_trig = npy.c_[p[t[:,0],0],p[t[:,1],0],p[t[:,2],0]]
        yy_trig = npy.c_[p[t[:,0],1],p[t[:,1],1],p[t[:,2],1]]
        
        
        if DATAT == 'P1d':
            zz = u[:3*nt].reshape(nt,3)
        if DATAT == 'P0':
            zz = npy.tile(u[:nt],(3,1)).T
        if DATAT == 'P1':
            zz = u[t]
        
        ii, jj, kk = npy.r_[:3*nt].reshape((nt, 3)).T
        fig.add_trace(go.Mesh3d(
            name = 'Trig values',
            x = xx_trig.ravel(), y = yy_trig.ravel(), z = u_height*zz.ravel(),
            i = ii, j = jj, k = kk, intensity = zz.ravel(), 
            colorscale = 'Rainbow',
            cmin = npy.min(u),
            # hoverinfo = 'skip',
            cmax = npy.max(u),
            intensitymode = 'vertex',
            lighting = dict(ambient = 1),
            contour_width = 1, contour_color = "#000000", contour_show = True,
            # xaxis = dict(range())
        ))
        
        # fig.add_trace(go.Contour(z=zz.flatten(), showscale=False))
        # fig.add_trace(go.Surface(x = p[:,0], 
        #                          y = p[:,1], 
        #                          z = u,
        #                          contours_z = dict(show = True, 
        #                                            usecolormap = True, 
        #                                            highlightcolor = "limegreen", 
        #                                            project_z = True)))
        
        xxx_trig = npy.c_[xx_trig,xx_trig[:,0],npy.nan*xx_trig[:,0]]
        yyy_trig = npy.c_[yy_trig,yy_trig[:,0],npy.nan*yy_trig[:,0]]
        zzz_trig = npy.c_[zz,zz[:,0],npy.nan*zz[:,0]]
        
        fig.add_trace(go.Scatter3d(name = 'Trig traces',
                                    mode = 'lines',
                                    x = xxx_trig.ravel(),
                                    y = yyy_trig.ravel(),
                                    z = u_height*zzz_trig.ravel(),
                                    line = go.scatter3d.Line(color = 'black', 
                                                            width = 1.5),
                                    showlegend = False))
        
        
        if DATAT == 'P1':
            x = p[:,0]; y = p[:,1]; z = u;
            xr = npy.linspace(x.min(), x.max(), 200); yr = npy.linspace(y.min(), y.max(), 200)
            xr, yr = npy.meshgrid(xr, yr)
            Z = griddata((x, y), z, (xr, yr) , method='cubic', fill_value=0)
            # Z = griddata((x, y), z, (xr, yr), fill_value=0)
            # print(u_height)
            # print(xr,yr,z)
            fig.add_trace(go.Surface(name = 'Isolines',
                                      x = xr[0],
                                      y = yr[:,0],
                                      z = Z, hidesurface = True,
                                      showlegend = None,
                                      showscale = False,
                                      contours_z = dict(show = True,
                                                        start = Z.min(),
                                                        end = 0.95*Z.max(),
                                                        size = (Z.max()-Z.min())/15,
                                                        width = 2,
                                                        # usecolormap = True,
                                                        project_z = True,
                                                        highlightcolor = "#FFFFFF",
                                                        usecolormap = False,
                                                        # highlightwidth = 16,
                                                        color = "black"
                                                        )
                                      ))
            # x = p[:,0]; y = p[:,1]; z = u;
            
            # print(x.shape,y.shape,z.shape)
            
            # fig.add_trace(go.Surface(name = 'whatever',
            #                           x = x,
            #                           y = y,
            #                           z = npy.c_[z,z],
            #                           hidesurface = False
            #                           )
            #               )
            # fig.add_contour(z=z)
            
            
        # fig.update_traces()
        
        return fig
        
        
        
    def __pdesurf_quad(self,fig,DATAT,u,u_height):
        nq = self.nq; p = self.p; q = self.q;
        
        xx_quad_1 = npy.c_[p[q[:,0],0],p[q[:,1],0],p[q[:,2],0]]
        yy_quad_1 = npy.c_[p[q[:,0],1],p[q[:,1],1],p[q[:,2],1]]

        xx_quad_2 = npy.c_[p[q[:,0],0],p[q[:,2],0],p[q[:,3],0]]
        yy_quad_2 = npy.c_[p[q[:,0],1],p[q[:,2],1],p[q[:,3],1]]

        xx_quad = npy.r_[xx_quad_1,xx_quad_2]
        yy_quad = npy.r_[yy_quad_1,yy_quad_2]
        
        if DATAT == 'Q1d':
            # zz = u[3*nt:].reshape(nq,4)
            zz = u[-4*nq:].reshape(nq,4)
        if DATAT == 'Q0':
            zz = npy.tile(u[-nq:],(4,1)).T
        
        zz_quad_1 = zz[:,:3]
        zz_quad_2 = npy.c_[zz[:,0],zz[:,2],zz[:,3]]
        zz_quad = npy.r_[zz_quad_1,zz_quad_2]
    
        ii, jj, kk = npy.r_[:3*nq+3*nq].reshape((nq+nq, 3)).T
        fig.add_trace(go.Mesh3d(
            name = 'Quad values',
            x = xx_quad.ravel(), y = yy_quad.ravel(), z = u_height*zz_quad.ravel(),
            i = ii, j = jj, k = kk, intensity = zz_quad.ravel(), 
            colorscale = 'Rainbow',
            cmin = npy.min(u),
            # hoverinfo = 'skip',
            cmax = npy.max(u),
            lighting = dict(ambient = 1),
            contour_width=2, contour_color="#000000", contour_show=True, 
        ))
        
        xxx_quad = npy.c_[p[q[:,0],0],p[q[:,1],0],p[q[:,2],0],p[q[:,3],0],p[q[:,0],0],npy.nan*p[q[:,0],0]]
        yyy_quad = npy.c_[p[q[:,0],1],p[q[:,1],1],p[q[:,2],1],p[q[:,3],1],p[q[:,0],1],npy.nan*p[q[:,0],1]]
        zzz_quad = npy.c_[zz,zz[:,0],npy.nan*zz[:,0]]
        
        fig.add_trace(go.Scatter3d(name = 'Quad traces',
                                   mode = 'lines',
                                   x = xxx_quad.ravel(),
                                   y = yyy_quad.ravel(),
                                   z = u_height*zzz_quad.ravel(),
                                   line = go.scatter3d.Line(color = 'black', width = 1.5),
                                   showlegend = False))
        return fig

def intersect2d(X, Y):
        """
        Function to find intersection of two 2D arrays.
        Returns index of rows in X that are common to Y.
        """
        # dims = X.max(0)+1
        # out = np.where(np.in1d(np.ravel_multi_index(X.T,dims),\
        #                np.ravel_multi_index(Y.T,dims)))[0]
        
        dims = X.max(0)+1
        X1D = npy.ravel_multi_index(X.T,dims)
        searched_valuesID = npy.ravel_multi_index(Y.T,dims)
        sidx = X1D.argsort()
        out = sidx[npy.searchsorted(X1D,searched_valuesID,sorter=sidx)]
        return out
























class mesh3:
    def __init__(self, p, e, f, t, r3d = npy.empty(0), r2d = npy.empty(0), r1d = npy.empty(0), identifications = npy.empty(0)):
        
        # erst t sortieren! ... macht einiges einfacher
        t = npy.c_[npy.sort(t[:,:4]),t[:,4]]


        edges_tets = npy.r_[npy.c_[t[:,0],t[:,1]],
                            npy.c_[t[:,0],t[:,2]],
                            npy.c_[t[:,0],t[:,3]],
                            npy.c_[t[:,1],t[:,2]],
                            npy.c_[t[:,1],t[:,3]],
                            npy.c_[t[:,2],t[:,3]]]

        faces_tets = npy.r_[npy.c_[t[:,1],t[:,2],t[:,3]],
                            npy.c_[t[:,0],t[:,2],t[:,3]],
                            npy.c_[t[:,0],t[:,1],t[:,3]],
                            npy.c_[t[:,0],t[:,1],t[:,2]]]

        mp_tet = 1/4*(p[t[:,0],:] + p[t[:,1],:] + p[t[:,2],:] + p[t[:,3],:])

        e_new = npy.sort(e[:,:2])
        f_new = npy.sort(f[:,:3])
            
        nt = t.shape[0]

        #############################################################################################################
        edges = npy.sort(edges_tets).astype(int)
        EdgesToVertices, je = npy.unique(edges, axis = 0, return_inverse = True)

        NoEdges = EdgesToVertices.shape[0]
        TetsToEdges = je[:6*nt].reshape(nt, 6, order = 'F').astype(npy.int64)
        BoundaryEdges = intersect2d(EdgesToVertices,e_new)
        # InteriorEdges = npy.setdiff1d(npy.arange(NoEdges),BoundaryEdges)

        EdgesToVertices = npy.c_[EdgesToVertices,npy.zeros(EdgesToVertices.shape[0],dtype = npy.int64)-1]
        EdgesToVertices[BoundaryEdges,2] = e[:,-1]
        #############################################################################################################



        #############################################################################################################
        faces = npy.sort(faces_tets).astype(int)
        FacesToVertices, je = npy.unique(faces, axis = 0, return_inverse = True)
        
        NoFaces = FacesToVertices.shape[0]
        TetsToFaces = je[:4*nt].reshape(nt, 4, order = 'F').astype(npy.int64)
        BoundaryFaces = intersect2d(FacesToVertices,f_new)
        # InteriorFaces = npy.setdiff1d(npy.arange(NoFaces),BoundaryFaces)

        FacesToVertices = npy.c_[FacesToVertices,npy.zeros(FacesToVertices.shape[0],dtype = npy.int64)-1]
        FacesToVertices[BoundaryFaces,3] = f[:,-1]
        
        indices = npy.unique(TetsToFaces, return_index=True)[1]
        DirectionFaces = -1+0*TetsToFaces.copy()
        DirectionFaces.ravel()[indices] = 1
        #############################################################################################################


        if (t.shape[1] == 5): # linear mapping
            t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]; t3 = t[:,3]
            C00 = p[t0,0]; C01 = p[t1,0]; C02 = p[t2,0]; C03 = p[t3,0];
            C10 = p[t0,1]; C11 = p[t1,1]; C12 = p[t2,1]; C13 = p[t3,1];
            C20 = p[t0,2]; C21 = p[t1,2]; C22 = p[t2,2]; C23 = p[t3,2];
            
            Fx = lambda x,y,z : C00*(1-x-y-z) +C01*x +C02*y +C03*z
            Fy = lambda x,y,z : C10*(1-x-y-z) +C11*x +C12*y +C13*z
            Fz = lambda x,y,z : C20*(1-x-y-z) +C21*x +C22*y +C23*z
            
            JF00 = lambda x,y,z : C01-C00
            JF01 = lambda x,y,z : C02-C00
            JF02 = lambda x,y,z : C03-C00
            
            JF10 = lambda x,y,z : C11-C10
            JF11 = lambda x,y,z : C12-C10
            JF12 = lambda x,y,z : C13-C10
            
            JF20 = lambda x,y,z : C21-C20
            JF21 = lambda x,y,z : C22-C20
            JF22 = lambda x,y,z : C23-C20
            
            
            f0 = f[:,0]; f1 = f[:,1]; f2 = f[:,2]
            B00 = p[f0,0]; B01 = p[f1,0]; B02 = p[f2,0];
            B10 = p[f0,1]; B11 = p[f1,1]; B12 = p[f2,1];
            B20 = p[f0,2]; B21 = p[f1,2]; B22 = p[f2,2];
            
            Bx = lambda x,y : B00*(1-x-y) +B01*x +B02*y
            By = lambda x,y : B10*(1-x-y) +B11*x +B12*y
            Bz = lambda x,y : B20*(1-x-y) +B21*x +B22*y
            
            JB00 = lambda x,y : B01-B00
            JB01 = lambda x,y : B02-B00
            
            JB10 = lambda x,y : B11-B10
            JB11 = lambda x,y : B12-B10
            
            JB20 = lambda x,y : B21-B20
            JB21 = lambda x,y : B22-B20
            

        detA = lambda x,y,z : +JF00(x,y,z)*(JF11(x,y,z)*JF22(x,y,z)-JF12(x,y,z)*JF21(x,y,z)) \
                              -JF01(x,y,z)*(JF10(x,y,z)*JF22(x,y,z)-JF12(x,y,z)*JF20(x,y,z)) \
                              +JF02(x,y,z)*(JF10(x,y,z)*JF21(x,y,z)-JF11(x,y,z)*JF20(x,y,z))

        iJF00 = lambda x,y,z:  1/detA(x,y,z)*(JF22(x,y,z)*JF11(x,y,z)-JF21(x,y,z)*JF12(x,y,z))
        iJF10 = lambda x,y,z: -1/detA(x,y,z)*(JF22(x,y,z)*JF10(x,y,z)-JF20(x,y,z)*JF12(x,y,z))
        iJF20 = lambda x,y,z:  1/detA(x,y,z)*(JF21(x,y,z)*JF10(x,y,z)-JF20(x,y,z)*JF11(x,y,z))

        iJF01 = lambda x,y,z: -1/detA(x,y,z)*(JF22(x,y,z)*JF01(x,y,z)-JF21(x,y,z)*JF02(x,y,z))
        iJF11 = lambda x,y,z:  1/detA(x,y,z)*(JF22(x,y,z)*JF00(x,y,z)-JF20(x,y,z)*JF02(x,y,z))
        iJF21 = lambda x,y,z: -1/detA(x,y,z)*(JF21(x,y,z)*JF00(x,y,z)-JF20(x,y,z)*JF01(x,y,z))

        iJF02 = lambda x,y,z:  1/detA(x,y,z)*(JF12(x,y,z)*JF01(x,y,z)-JF11(x,y,z)*JF02(x,y,z))
        iJF12 = lambda x,y,z: -1/detA(x,y,z)*(JF12(x,y,z)*JF00(x,y,z)-JF10(x,y,z)*JF02(x,y,z))
        iJF22 = lambda x,y,z:  1/detA(x,y,z)*(JF11(x,y,z)*JF00(x,y,z)-JF10(x,y,z)*JF01(x,y,z))


        detB = lambda x,y : (JB00(x,y)**2 + JB10(x,y)**2 + JB20(x,y)**2)\
                           *(JB01(x,y)**2 + JB11(x,y)**2 + JB21(x,y)**2)\
                           -(JB00(x,y)*JB01(x,y) + JB10(x,y)*JB11(x,y) + JB20(x,y)*JB21(x,y))
                           
        iJB00 = lambda x,y:  1/detB(x,y)*(+JB00(x,y)*JB11(x,y)**2 \
                                          +JB00(x,y)*JB21(x,y)**2 \
                                          -JB01(x,y)*JB10(x,y)*JB11(x,y)\
                                          -JB01(x,y)*JB20(x,y)*JB21(x,y))
        
        iJB01 = lambda x,y:  1/detB(x,y)*(+JB01(x,y)*JB10(x,y)**2 \
                                          +JB10(x,y)*JB20(x,y)**2 \
                                          -JB00(x,y)*JB10(x,y)*JB11(x,y)\
                                          -JB00(x,y)*JB20(x,y)*JB21(x,y))
        
        iJB10 = lambda x,y:  1/detB(x,y)*(+JB10(x,y)*JB01(x,y)**2 \
                                          +JB10(x,y)*JB21(x,y)**2 \
                                          -JB00(x,y)*JB01(x,y)*JB11(x,y)\
                                          -JB11(x,y)*JB20(x,y)*JB21(x,y))
        
        iJB11 = lambda x,y:  1/detB(x,y)*(+JB11(x,y)*JB00(x,y)**2 \
                                          +JB11(x,y)*JB20(x,y)**2 \
                                          -JB00(x,y)*JB01(x,y)*JB10(x,y)\
                                          -JB10(x,y)*JB20(x,y)*JB21(x,y))
        
        iJB20 = lambda x,y:  1/detB(x,y)*(+JB20(x,y)*JB01(x,y)**2 \
                                          +JB20(x,y)*JB11(x,y)**2 \
                                          -JB00(x,y)*JB01(x,y)*JB21(x,y)\
                                          -JB10(x,y)*JB11(x,y)*JB21(x,y))
        
        iJB21 = lambda x,y:  1/detB(x,y)*(+JB21(x,y)*JB00(x,y)**2 \
                                          +JB21(x,y)*JB10(x,y)**2 \
                                          -JB00(x,y)*JB01(x,y)*JB20(x,y)\
                                          -JB20(x,y)*JB10(x,y)*JB11(x,y))

        #############################################################################################################

        self.EdgesToVertices = EdgesToVertices
        self.TetsToEdges = TetsToEdges
        self.NoEdges = NoEdges
        
        self.FacesToVertices = FacesToVertices
        self.TetsToFaces = TetsToFaces
        self.DirectionFaces = DirectionFaces
        self.NoFaces = NoFaces
        
        self.BoundaryEdges_Region = e[:,-1]
        self.Boundary_Edges = BoundaryEdges
        self.Boundary_NoEdges = BoundaryEdges.shape[0]
        
        self.BoundaryFaces_Region = f[:,-1]
        self.Boundary_Faces= BoundaryFaces
        self.Boundary_NoFaces = BoundaryFaces.shape[0]

        self.p = p; self.np = p.shape[0]
        self.e = npy.c_[e_new,e[:,-1]]; self.ne = e_new.shape[0]
        self.f = npy.c_[f_new,f[:,-1]]; self.nf = f_new.shape[0]
        # self.t = t[:,npy.r_[:3,-1]]; self.nt = nt
        self.t = t; self.nt = nt
        
        self.regions_3d = list(r3d)
        self.regions_2d = list(r2d)
        self.regions_1d = list(r1d)
        self.identifications = identifications
        
        self.FEMLISTS = {}
        #############################################################################################################
        
        self.Fx = Fx; self.Fy = Fy; self.Fz = Fz;
        self.detA = detA
        self.mp_tet = mp_tet
        
        self.JF00 = JF00; self.JF10 = JF10; self.JF20 = JF20;
        self.JF01 = JF01; self.JF11 = JF11; self.JF21 = JF21;
        self.JF02 = JF02; self.JF12 = JF12; self.JF22 = JF22;
        
        self.iJF00 = iJF00; self.iJF10 = iJF10; self.iJF20 = iJF20;
        self.iJF01 = iJF01; self.iJF11 = iJF11; self.iJF21 = iJF21;
        self.iJF02 = iJF02; self.iJF12 = iJF12; self.iJF22 = iJF22;
        
        self.Bx = Bx; self.By = By;
        self.detB = detB
        
        self.JB00 = JB00; self.JB10 = JB10;
        self.JB01 = JB01; self.JB11 = JB11;
        
        self.iJB00 = iJB00; self.iJB10 = iJB10;
        self.iJB01 = iJB01; self.iJB11 = iJB11;
        
        #############################################################################################################

    def __repr__(self):
        return f"np:{self.np}, nt:{self.nt}, nf:{self.nf}, ne:{self.ne}, nf_all:{self.NoFaces}, ne_all:{self.NoEdges}"
    
    def from_netgen(self,geoOCCmesh):
        npoints2D = geoOCCmesh.Elements2D().NumPy()['np'].max()
        npoints3D = geoOCCmesh.Elements3D().NumPy()['np'].max()

        p = geoOCCmesh.Coordinates()

        t = npy.c_[geoOCCmesh.Elements3D().NumPy()['nodes'].astype(npy.uint64)[:,:npoints3D],
                   geoOCCmesh.Elements3D().NumPy()['index'].astype(npy.uint64)]-1

        f = npy.c_[geoOCCmesh.Elements2D().NumPy()['nodes'].astype(npy.uint64)[:,:npoints2D],
                   geoOCCmesh.Elements2D().NumPy()['index'].astype(npy.uint64)]-1

        e = npy.c_[geoOCCmesh.Elements1D().NumPy()['nodes'].astype(npy.uint64)[:,:((npoints2D+1)//2)],
                   geoOCCmesh.Elements1D().NumPy()['index'].astype(npy.uint64)]-1

        max_ed_index = geoOCCmesh.Elements1D().NumPy()['index'].astype(npy.uint64).max()
        max_bc_index = geoOCCmesh.Elements2D().NumPy()['index'].astype(npy.uint64).max()
        max_rg_index = geoOCCmesh.Elements3D().NumPy()['index'].astype(npy.uint64).max()

        regions_1d_np = []
        for i in range(max_ed_index):
            regions_1d_np += [geoOCCmesh.GetCD2Name(i)]
            
        regions_2d_np = []
        for i in range(max_bc_index):
            regions_2d_np += [geoOCCmesh.GetBCName(i)]

        regions_3d_np = []
        for i in range(max_rg_index):
            regions_3d_np += [geoOCCmesh.GetMaterial(i+1)]

        identifications = npy.array(geoOCCmesh.GetIdentifications())
        
        return p, e, f, t, regions_3d_np, regions_2d_np, regions_1d_np, identifications
    
    @classmethod
    def netgen(cls, geoOCCmesh):
        p, e, f, t, regions_3d_np, regions_2d_np, regions_1d_np, identifications = cls.from_netgen(cls,geoOCCmesh)
        return cls(p, e, f, t, regions_3d_np, regions_2d_np, regions_1d_np, identifications)
    
    def pdesurf(self, u, faces, **kwargs):
        
        cmin = npy.min(u)
        cmax = npy.max(u)
        
        if 'cmin' in kwargs.keys():
            cmin = kwargs['cmin']
        if 'cmax' in kwargs.keys():
            cmax = kwargs['cmax']
            
        ind_regions = getIndices(self.regions_2d, faces)
        indices = npy.in1d(self.f[:,-1],ind_regions)
        
        nt = self.nt; p = self.p; t = self.t[:,:3]; np = self.np; 
        f = self.f[indices,:3]; nf = f.shape[0]
        
        xx_trig = npy.c_[p[f[:,0],0],p[f[:,1],0],p[f[:,2],0]]
        yy_trig = npy.c_[p[f[:,0],1],p[f[:,1],1],p[f[:,2],1]]
        zz_trig = npy.c_[p[f[:,0],2],p[f[:,1],2],p[f[:,2],2]]
        
        # if u.size == 3*nf:
        #     zz = u[:3*nf].reshape(nf,3)
        # if u.size == nt:
        #     zz = npy.tile(u[:nt],(3,1)).T
        if u.size == np:
            zz = u[f]
        
        fig = go.Figure()
        
        ii, jj, kk = npy.r_[:3*nf].reshape((nf, 3)).T
        fig.add_trace(go.Mesh3d(
            name = 'Trig values',
            x = xx_trig.ravel(),
            y = yy_trig.ravel(),
            z = zz_trig.ravel(),
            i = ii, j = jj, k = kk, intensity = zz.ravel(),
            colorscale = 'Jet',
            intensitymode = 'vertex',
            lighting = dict(ambient = 1),
            contour_width = 1, contour_color = "#000000", contour_show = True,
            # xaxis = dict(range())
        ))

        xxx_trig = npy.c_[xx_trig,xx_trig[:,0],npy.nan*xx_trig[:,0]]
        yyy_trig = npy.c_[yy_trig,yy_trig[:,0],npy.nan*yy_trig[:,0]]
        zzz_trig = npy.c_[zz_trig,zz_trig[:,0],npy.nan*zz_trig[:,0]]

        fig.add_trace(go.Scatter3d(name = 'Trig traces',
                                    mode = 'lines',
                                    x = xxx_trig.ravel(),
                                    y = yyy_trig.ravel(),
                                    z = zzz_trig.ravel(),
                                    line = go.scatter3d.Line(color = 'black', width = 1.5),
                                    showlegend = False))

        color_scales_3D = ['Blackbody','Bluered','Blues','Cividis','Earth','Electric','Greens','Greys','Hot','Jet','Picnic','Portland','Rainbow','RdBu','Reds','Viridis','YlGnBu','YlOrRd']
        list_color_scales = list(map(lambda x: dict(args=["colorscale", x],label=x,method="restyle"),color_scales_3D))
                            
        fig.update_layout(
        updatemenus = [
            dict(
                buttons = list_color_scales,
                direction = "down",
                showactive = False,
                xanchor = "left",
                yanchor = "top"),
            dict(
                buttons = list([
                    dict(
                        args = ["scene.camera.projection.type", "orthographic"],
                        label = "Orthographic",
                        method = "relayout"
                    ),
                    dict(
                        args = ["scene.camera.projection.type", "perspective"],
                        label = "Perspective",
                        method = "relayout"
                    ),
                ]),
                direction="down",
                showactive=True,
                xanchor="left",
                yanchor="bottom")],
                      margin_l=0,
                      margin_t=0,
                      margin_r=0,
                      margin_b=0,)
        
        
        fig.show()
        return fig
    
    
    # def getIndices(self, liste, name):
    #     regions = npy.char.split(name,',').tolist()
    #     ind = npy.empty(shape=(0,),dtype=npy.int64)
    #     for k in regions:
    #         if k[0] == '*':
    #             n = npy.flatnonzero(npy.char.find(liste,k[1:])!=-1)
    #         else:
    #             n = npy.flatnonzero(npy.char.equal(liste,k))
    #         ind = npy.append(ind,n,axis=0)
    #     return npy.unique(ind)
    
    
# def refinemesh(p,e,t):
    
    
    
#     pn = 1/2*(p[EdgesToVertices[:,0],:]+
#               p[EdgesToVertices[:,1],:])
#     p_new = npy.r_[p,pn]
    
#     tn = np + TriangleToEdges
#     t_new = npy.r_[npy.c_[t[:,0],tn[:,2],tn[:,1],RegionsT],
#                    npy.c_[t[:,1],tn[:,0],tn[:,2],RegionsT],
#                    npy.c_[t[:,2],tn[:,1],tn[:,0],RegionsT],
#                    npy.c_[    tn[:,0],tn[:,1],tn[:,2],RegionsT]].astype(npy.int64)
#     bn = MESH.np + MESH.Boundary.Edges
#     e_new = npy.r_[npy.c_[e[:,0],bn,Boundary.Region],
#                    npy.c_[bn,e[:,1],Boundary.Region]].astype(npy.int64)
    
    
    
#     return p_new,e_new,t_new

# DEBUGGING
# if __name__ == "__main__":
#     from pde.petq_from_gmsh import *
#     p,e,t,q = petq_from_gmsh(filename='mesh_new.geo',hmax=0.8)
#     mesh = create_mesh(p,e,t,q)
#     mesh.pdemesh()
#     print('dsa')
