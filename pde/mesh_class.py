
import numpy as npy # npy becauswe np is number of points... dont judge me :)
npy.set_printoptions(edgeitems=10, linewidth = 1000000)
# npy.set_printoptions(threshold = npy.inf)
# npy.set_printoptions(linewidth = npy.inf)
npy.set_printoptions(precision=2)

# import pandas as pd
import plotly.graph_objects as go
# import plotly.colors as plyc
from scipy.interpolate import griddata
from . import lists as femlists
import numba as jit
# from .tools import *
# from .tools import unique_rows

# import plotly.figure_factory as ff


# import matplotlib.pyplot as plt
# import plotly.express as px
# import pandas as pd

# from plotly_resampler import register_plotly_resampler

# Call the register function once and all Figures/FigureWidgets will be wrapped
# according to the register_plotly_resampler its `mode` argument
# register_plotly_resampler(mode='auto')

class mesh:
    # @profile
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
            
        e_new = npy.sort(e[:,0:2])
        
        nt = t.shape[0]
        nq = q.shape[0]
        np = p.shape[0]
        ne = e_new.shape[0]
        
        #############################################################################################################
        edges = npy.r_[npy.sort(edges_trigs),
                       npy.sort(edges_quads)].astype(int)
        EdgesToVertices, je = npy.unique(edges, axis = 0, return_inverse = True)
        # EdgesToVertices, je = unique_rows(edges, return_inverse = True)
        
        NoEdges = EdgesToVertices.shape[0]
        TriangleToEdges = je[0:3*nt].reshape(nt,3, order = 'F').astype(npy.int64)
        QuadToEdges = je[3*nt:].reshape(nq,4, order = 'F')
        
        BoundaryEdges = intersect2d(EdgesToVertices,e_new)
        #############################################################################################################
        
        self.EdgesToVertices = EdgesToVertices
        self.TriangleToEdges = TriangleToEdges
        self.QuadToEdges = QuadToEdges
        self.NoEdges = NoEdges
        self.EdgeDirectionTrig = EdgeDirectionTrig
        self.EdgeDirectionQuad = EdgeDirectionQuad
        self.Boundary_Region = e[:,2]
        self.Boundary_Edges = BoundaryEdges
        self.Boundary_NoEdges = BoundaryEdges.shape[0]

        self.p = p; self.np = np
        self.e = e_new; self.ne = ne
        self.t = t; self.nt = nt
        self.q = q; self.nq = nq
        self.mp = npy.r_[mp_trig,mp_quad]
        
        # if t.shape[1]>3: self.RegionsT = t[:,3]
        # if q.shape[1]>4: self.RegionsQ = q[:,4]
        
        self.FEMLISTS = {}
        
    def __repr__(self):
        return f"np:{self.np}, nt:{self.nt}, nq:{self.nq}, ne:{self.ne}, ne_all:{self.NoEdges}"
    
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
        
    def refinemesh(self):
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
        
        self.__init__(p_new,e_new,t_new,q_new)
        self.FEMLISTS = {} # reset fem lists, cuz new mesh
        # self.delattr("Boundary_EdgeOrientation")
        
        if hasattr(self, "Boundary_EdgeOrientation"):
            del self.Boundary_EdgeOrientation

        
        print('Generated refined mesh with ' + str(p_new.shape[0]) + ' points, ' 
                                             + str(e_new.shape[0]) + ' boundary edges, ' 
                                             + str(t_new.shape[0]) + ' triangles, ' 
                                             + str(q_new.shape[0]) + ' quadrilaterals.')
        # return p_new,e_new,t_new
        
    
    def refine(self,f):
        # if f.size != 3*self.nt:
        #     print('Wrong size! must be P1-discontinuous')
        #     return        
        return npy.r_[f,f,f,f]
    
    def trianglesFromPoints(self, points):
        return npy.unique(npy.where(npy.in1d(self.t[:,0:3].flatten(),points))[0]//3)
        
    def __ismember(self,a_vec, b_vec):
        """ MATLAB equivalent ismember function """
        bool_ind = npy.isin(a_vec,b_vec)
        common = a_vec[bool_ind]
        common_unique, common_inv  = npy.unique(common, return_inverse=True)     # common = common_unique[common_inv]
        b_unique, b_ind = npy.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
        common_ind = b_ind[npy.isin(b_unique, common_unique, assume_unique=True)]
        return bool_ind, common_ind[common_inv]
    
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
        
        x = npy.c_[p[t[:,0],0],p[t[:,1],0],p[t[:,2],0],p[t[:,0],0],npy.nan*p[t[:,0],0]].flatten()#.tolist()
        y = npy.c_[p[t[:,0],1],p[t[:,1],1],p[t[:,2],1],p[t[:,0],1],npy.nan*p[t[:,0],1]].flatten()#.tolist()
                        
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
                                   x=xxx_trig.flatten(),
                                   y=yyy_trig.flatten(),
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
                          # scene = scene,
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
                    showactive = True,
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
            x = xx_trig.flatten(), y = yy_trig.flatten(), z = u_height*zz.flatten(),
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
                                    z = u_height*zzz_trig.flatten(),
                                    line = go.scatter3d.Line(color = 'black', 
                                                            width = 1.5),
                                    showlegend = False))
        
        
        if DATAT == 'P1':
            x = p[:,0]; y = p[:,1]; z = u;
            xr = npy.linspace(x.min(), x.max(), 200); yr = npy.linspace(y.min(), y.max(), 100)
            xr, yr = npy.meshgrid(xr, yr)
            Z = griddata((x, y), z, (xr, yr) , method='cubic', fill_value=0)
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
                                                        end = Z.max(),
                                                        size = (Z.max()-Z.min())/30,
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
            x = xx_quad.flatten(), y = yy_quad.flatten(), z = u_height*zz_quad.flatten(),
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
                                   z = u_height*zzz_quad.flatten(),
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