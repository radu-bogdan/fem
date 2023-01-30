
import numpy as npy # npy becauswe np is number of points... dont judge me :)
npy.set_printoptions(edgeitems=30, linewidth = 1000000)
npy.set_printoptions(threshold = npy.inf)
npy.set_printoptions(linewidth = npy.inf)
npy.set_printoptions(precision=2)

import plotly.graph_objects as go
# import plotly.colors as plyc
from scipy.interpolate import griddata
from . import lists as femlists

# import plotly.figure_factory as ff


# import matplotlib.pyplot as plt
# import plotly.express as px
# import pandas as pd

class mesh:
    def __init__(self, p,e,t,q):
        
        if t.size != 0:
            edges_trigs = npy.r_[npy.c_[t[:,1],t[:,2]],
                                 npy.c_[t[:,2],t[:,0]],
                                 npy.c_[t[:,0],t[:,1]]]
            EdgeDirectionTrig = npy.sign(npy.c_[t[:,1]-t[:,2],
                                                t[:,2]-t[:,0],
                                                t[:,0]-t[:,1]].astype(int))*(-1)
            mp_trig = 1/3*(p[t[:,0],:] + p[t[:,1],:] + p[t[:,2],:])
        else:
            edges_trigs = npy.array([], dtype=npy.int64).reshape(0,2)
            EdgeDirectionTrig= npy.array([], dtype=npy.int64).reshape(0,3)
            mp_trig = npy.array([], dtype=npy.int64).reshape(0,2)
        

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
            edges_quads = npy.array([], dtype=npy.int64).reshape(0,2)
            EdgeDirectionQuad= npy.array([], dtype=npy.int64).reshape(0,4)
            mp_quad = npy.array([], dtype=npy.int64).reshape(0,2)

        e_new = npy.sort(e[:,0:2])
        nt = t.shape[0]
        nq = q.shape[0]
        np = p.shape[0]
        ne = e_new.shape[0]
        
        #############################################################################################################
        edges = npy.r_[npy.sort(edges_trigs),npy.sort(edges_quads)].astype(int)
        EdgesToVertices, je = npy.unique(edges,axis=0, return_inverse=True)

        NoEdges = EdgesToVertices.shape[0]
        TriangleToEdges = je[0:3*nt].reshape(nt,3, order='F')
        QuadToEdges = je[3*nt:].reshape(nq,4, order='F')
        #############################################################################################################
        
        #############################################################################################################
        # Need this so we can find the indices of the boundary edges in the global edge list
        BoundaryEdges2 = npy.argwhere(npy.bincount(je)==1)[:,0]
        _,je_new = npy.unique(e_new, axis=0, return_inverse=True)
        BoundaryEdges = BoundaryEdges2[je_new]
        e_new = EdgesToVertices[BoundaryEdges,:]
        #############################################################################################################

        #############################################################################################################
        loc_trig,index_trig = self.__ismember(TriangleToEdges,BoundaryEdges)
        loc_quad,index_quad = self.__ismember(QuadToEdges,BoundaryEdges)

        indices_boundary = npy.r_[index_trig,index_quad]
        direction_boundary = npy.r_[EdgeDirectionTrig[loc_trig],EdgeDirectionQuad[loc_quad]]
        b = npy.argsort(indices_boundary)
        #############################################################################################################

        #############################################################################################################
        self._Boundary_Region = e[:,2]
        self._Boundary_p_index = npy.unique(e)
        self._Boundary_np = self._Boundary_p_index.size
        self._Boundary_Edges = BoundaryEdges
        self._Boundary_NoEdges = BoundaryEdges.shape[0]
        self._Boundary_EdgeOrientation = direction_boundary[b]
        #############################################################################################################

        #############################################################################################################
        self.EdgesToVertices = EdgesToVertices
        self.TriangleToEdges = TriangleToEdges
        self.QuadToEdges = QuadToEdges
        self.NoEdges = NoEdges

        self.EdgeDirectionTrig = EdgeDirectionTrig
        self.EdgeDirectionQuad = EdgeDirectionQuad

        self.p = p
        self.e = e_new
        self.t = t[:,0:3]
        self.q = q[:,0:4]
        
        if t.shape[1]>3:
            self.RegionsT = t[:,3]
        if q.shape[1]>4:
            self.RegionsQ = q[:,4]

        self.np = np
        self.ne = ne
        self.nt = nt
        self.nq = nq
        self.mp = npy.r_[mp_trig,mp_quad]
        #############################################################################################################
    
        Edges_Triangle_Mix = npy.unique(TriangleToEdges)
        Edges_Quad_Mix = npy.unique(QuadToEdges)
        self._Lists_InterfaceTriangleQuad = npy.intersect1d(Edges_Triangle_Mix,Edges_Quad_Mix)
        self._Lists_JustTrig = npy.setdiff1d(Edges_Triangle_Mix,self._Lists_InterfaceTriangleQuad)
        self._Lists_JustQuad = npy.setdiff1d(Edges_Quad_Mix,self._Lists_InterfaceTriangleQuad)
        self._Lists_TrigBoundaryEdges = npy.intersect1d(self._Lists_JustTrig,BoundaryEdges)

        loc,_ = self.__ismember(QuadToEdges,self._Lists_InterfaceTriangleQuad)
        QuadsAtTriangleInterface = npy.argwhere(loc)[:,0]
        QuadLayerEdges = npy.unique(QuadToEdges[QuadsAtTriangleInterface,:])


        self._Lists_QuadLayerEdges = QuadLayerEdges
        self._Lists_QuadBoundaryEdges = npy.intersect1d(self._Lists_JustQuad,BoundaryEdges)
        self._Lists_QuadsAtTriangleInterface = QuadsAtTriangleInterface

        self.Boundary = self.boundary(self)
        self.Lists = self.lists(self)
        
        #############################################################################################################
        
        regions_to_points = npy.empty(shape = [0,2],dtype = 'uint64')
        if t.shape[1]>3:
            for i in range(max(self.RegionsT)):
                indices = npy.unique(self.t[npy.argwhere(self.RegionsT == (i+1))[:,0],:])            
                vtr = npy.c_[(i+1)*npy.ones(shape = indices.shape,dtype = 'uint64'), indices]
                regions_to_points = npy.r_[regions_to_points, vtr]
            
        self.RegionsToPoints = regions_to_points

    class boundary:
        def __init__(self,parent):
            self.Region = parent._Boundary_Region
            self.p_index = parent._Boundary_p_index
            self.np = parent._Boundary_p_index
            self.Edges = parent._Boundary_Edges
            self.NoEdges = parent._Boundary_NoEdges
            self.EdgeOrientation = parent._Boundary_EdgeOrientation
            self.TrigEdges = parent._Lists_TrigBoundaryEdges
            self.QuadEdges = parent._Lists_QuadBoundaryEdges

    class lists:
        def __init__(self,parent):
            self.InterfaceTriangleQuad = parent._Lists_InterfaceTriangleQuad
            self.JustTrig = parent._Lists_JustTrig
            self.JustQuad = parent._Lists_JustQuad
            self.QuadLayerEdges = parent._Lists_QuadLayerEdges
            self.QuadsAtTriangleInterface = parent._Lists_QuadsAtTriangleInterface
        
    def makeFemLists(self):
        self.FEMLISTS = femlists.lists(self)
    
    def refinemesh(self):
        pn = 1/2*(self.p[self.EdgesToVertices[:,0],:]+
                  self.p[self.EdgesToVertices[:,1],:])
        p_new = npy.r_[self.p,pn]
        
        tn = self.np + self.TriangleToEdges
        t_new = npy.r_[npy.c_[self.t[:,0],tn[:,2],tn[:,1],self.RegionsT],
                       npy.c_[self.t[:,1],tn[:,0],tn[:,2],self.RegionsT],
                       npy.c_[self.t[:,2],tn[:,1],tn[:,0],self.RegionsT],
                       npy.c_[    tn[:,0],tn[:,1],tn[:,2],self.RegionsT]].astype(npy.uint64)
        bn = self.np + self.Boundary.Edges
        e_new = npy.r_[npy.c_[self.e[:,0],bn,self.Boundary.Region],
                       npy.c_[bn,self.e[:,1],self.Boundary.Region]].astype(npy.uint64)
        
        return p_new,e_new,t_new
        
        
    def __ismember(self,a_vec, b_vec):
        """ MATLAB equivalent ismember function """
        bool_ind = npy.isin(a_vec,b_vec)
        common = a_vec[bool_ind]
        common_unique, common_inv  = npy.unique(common, return_inverse=True)     # common = common_unique[common_inv]
        b_unique, b_ind = npy.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
        common_ind = b_ind[npy.isin(b_unique, common_unique, assume_unique=True)]
        return bool_ind, common_ind[common_inv]
    
    def pdemesh(self,dpi=500,info=0,border=0.5):

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

        p = self.p; t = self.t

        x = npy.c_[p[t[:,0],0],p[t[:,1],0],p[t[:,2],0],p[t[:,0],0],npy.nan*p[t[:,0],0]].flatten().tolist()
        y = npy.c_[p[t[:,0],1],p[t[:,1],1],p[t[:,2],1],p[t[:,0],1],npy.nan*p[t[:,0],1]].flatten().tolist()

        fig.add_trace(go.Scattergl(mode='lines', 
                                    name='TrigTraces',
                                    x=x, 
                                    y=y, 
                                    line=dict(color='blue', 
                                                width=2)))
        if info == 1:
            fig.add_trace(go.Scattergl(mode='text+markers',
                                    name='Triangles',
                                    x = 1/3*(p[t[:,0],0]+p[t[:,1],0]+p[t[:,2],0]),
                                    y = 1/3*(p[t[:,0],1]+p[t[:,1],1]+p[t[:,2],1]),
                                    text = list(range(0, t.shape[0])),
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

    def __pdemesh_quad(self,fig,info):

        p = self.p; q = self.q

        x = npy.c_[p[q[:,0],0],p[q[:,1],0],p[q[:,2],0],p[q[:,3],0],p[q[:,0],0],npy.nan*p[q[:,0],0]].flatten().tolist()
        y = npy.c_[p[q[:,0],1],p[q[:,1],1],p[q[:,2],1],p[q[:,3],1],p[q[:,0],1],npy.nan*p[q[:,0],1]].flatten().tolist()
        
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
    
    
    
    def pdesurf_hybrid(self,TrigQuadDict,u):
        
        DATAT = TrigQuadDict.get('trig')
        DATAQ = TrigQuadDict.get('quad')
        controls = TrigQuadDict.get('controls')
        
        fig = go.Figure()
        
        if self.t.shape[0]!=0:
            fig = self.__pdesurf_trig(fig,DATAT,u)
        
        if self.q.shape[0]!=0:
            fig = self.__pdesurf_quad(fig,DATAQ,u)
        
        # camera = dict(up = dict(x = 1, y = 0., z = 0),
        #               eye = dict(x = 0, y = 0, z = 2*max(u)))
        camera = dict(eye = dict(x = 0, y = -1e-5, z = 1e10))
        ratio = (max(self.p[:,0])-min(self.p[:,0]))/(max(self.p[:,1])-min(self.p[:,1]))
        fig.update_layout(scene = dict(aspectratio = dict(x = ratio, y = 1, z = 1),
                                       xaxis = dict(showspikes = False),
                                       yaxis = dict(showspikes = False),
                                       zaxis = dict(showspikes = False)),
                          scene_camera = camera,
                          legend = dict(yanchor = "top",
                                        y = 1,
                                        xanchor = "left",
                                        x = -0.2,
                                        bgcolor = "LightSteelBlue",
                                        bordercolor = "Black",
                                        borderwidth = 2),
                          # yaxis=dict(
                          #       range=[0, 300]
                          #   ),
                          #   xaxis=dict(
                          #       range=[0, 300]
                             # )
                          )
        
        fig.layout.scene.camera.projection.type = "orthographic"
        fig.update_traces(showlegend = True)
        
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
                    showactive = True,
                    x = 0,
                    xanchor = "right",
                    y = 1.2,
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
                    x=0.2,
                    xanchor="right",
                    y=1.2,
                    yanchor="top")])
        
        
        return fig
            

    def __pdesurf_trig(self,fig,DATAT,u):
        nt = self.nt; p = self.p; t = self.t;
        
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
            x = xx_trig.flatten(), y = yy_trig.flatten(), z = zz.flatten(),
            i = ii, j = jj, k = kk, intensity = zz.flatten(), 
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
                                   x = xxx_trig.flatten(),
                                   y = yyy_trig.flatten(),
                                   z = zzz_trig.flatten(),
                                   line = go.scatter3d.Line(color = 'black', 
                                                            width = 1.5),
                                   showlegend = False))
        
        
        if DATAT == 'P1':
            x = p[:,0]; y = p[:,1]; z = u;
            xr = npy.linspace(x.min(), x.max(), 200); yr = npy.linspace(y.min(), y.max(), 100)
            xr, yr = npy.meshgrid(xr, yr)
            Z = griddata((x, y), z, (xr, yr) , method='cubic')

            fig.add_trace(go.Surface(name = 'Isolines',
                                     x = xr[0],
                                     y = yr[:,0],
                                     z = Z, hidesurface = True,
                                     showlegend = None,
                                     showscale = False,
                                     contours_z = dict(show = True,
                                                       start = Z.min(),
                                                       end = Z.max(),
                                                       size = (Z.max()-Z.min())/30,
                                                       width = 1,
                                                       # usecolormap = True,
                                                       # project_z = True,
                                                       highlightcolor = "#FFFFFF",
                                                       usecolormap = False,
                                                       # highlightwidth = 16,
                                                       color = "white"
                                                       )
                                     ))
        
        fig.update_traces()
        
        return fig
        
        
        
    def __pdesurf_quad(self,fig,DATAT,u):
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
            x = xx_quad.flatten(), y = yy_quad.flatten(), z = zz_quad.flatten(),
            i = ii, j = jj, k = kk, intensity = zz_quad.flatten(), 
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
                                   x = xxx_quad.flatten(),
                                   y = yyy_quad.flatten(),
                                   z = zzz_quad.flatten(),
                                   line = go.scatter3d.Line(color = 'black', width = 1.5),
                                   showlegend = False))
        return fig

# DEBUGGING
# if __name__ == "__main__":
#     from pde.petq_from_gmsh import *
#     p,e,t,q = petq_from_gmsh(filename='mesh_new.geo',hmax=0.8)
#     mesh = create_mesh(p,e,t,q)
#     mesh.pdemesh()
#     print('dsa')