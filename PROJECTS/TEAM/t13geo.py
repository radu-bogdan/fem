import ngsolve as ng
import netgen.occ as occ
# import netgen.gui
from netgen.webgui import Draw as DrawGeo

box1 = occ.Box(occ.Pnt(-100,-100,-50), occ.Pnt(100,100,50))
box2 = occ.Box(occ.Pnt(-75,-75,-50), occ.Pnt(75,75,50))
    
##########################################################################
# Rounding corners ...
##########################################################################

corner1_ext = occ.Box(occ.Pnt(75,75,-50), occ.Pnt(100,100,50))
cyl1_ext = occ.Cylinder(occ.Pnt(75,75,-50), occ.Z, r=25, h=100)
corner1_int = occ.Box(occ.Pnt(50,50,-50), occ.Pnt(75,75,50))
cyl1_int = occ.Cylinder(occ.Pnt(50,50,-50), occ.Z, r=25, h=100)
corner1_int = corner1_int-cyl1_int; corner1_ext = corner1_ext-cyl1_ext

corner2_ext = occ.Box(occ.Pnt(-100,-100,-50), occ.Pnt(-75,-75,50))
cyl2_ext = occ.Cylinder(occ.Pnt(-75,-75,-50), occ.Z, r=25, h=100)
corner2_int = occ.Box(occ.Pnt(-75,-75,-50), occ.Pnt(-50,-50,50))
cyl2_int = occ.Cylinder(occ.Pnt(-50,-50,-50), occ.Z, r=25, h=100)
corner2_int = corner2_int-cyl2_int; corner2_ext = corner2_ext-cyl2_ext

corner3_ext = occ.Box(occ.Pnt(75,-75,-50), occ.Pnt(100,-100,50))
cyl3_ext = occ.Cylinder(occ.Pnt(75,-75,-50), occ.Z, r=25, h=100)
corner3_int = occ.Box(occ.Pnt(50,-50,-50), occ.Pnt(75,-75,50))
cyl3_int = occ.Cylinder(occ.Pnt(50,-50,-50), occ.Z, r=25, h=100)
corner3_int = corner3_int-cyl3_int; corner3_ext = corner3_ext-cyl3_ext

corner4_ext = occ.Box(occ.Pnt(-75,75,-50), occ.Pnt(-100,100,50))
cyl4_ext = occ.Cylinder(occ.Pnt(-75,75,-50), occ.Z, r=25, h=100)
corner4_int = occ.Box(occ.Pnt(-50,50,-50), occ.Pnt(-75,75,50))
cyl4_int = occ.Cylinder(occ.Pnt(-50,50,-50), occ.Z, r=25, h=100)
corner4_int = corner4_int-cyl4_int; corner4_ext = corner4_ext-cyl4_ext

##########################################################################
# Adding the steel parts
##########################################################################

coil = (box1-box2)+corner1_int-corner1_ext+corner2_int-corner2_ext+corner3_int-corner3_ext+corner4_int-corner4_ext

mid_steel = occ.Box(occ.Pnt(-1.6,-25,-64.2),occ.Pnt(1.6,25,64.2))

r_steel1 = occ.Box(occ.Pnt(1.6+0.5,15,50+10-3.2),occ.Pnt(1.6+120.5,65,50+10))
r_steel2 = occ.Box(occ.Pnt(1.6+0.5,15,-(50+10-3.2)),occ.Pnt(1.6+120.5,65,-(50+10)))
r_steel3 = occ.Box(occ.Pnt(1.6+120.5-3.2,15,50+10-3.2),occ.Pnt(1.6+120.5,65,-(50+10)))
r_steel = r_steel1 + r_steel2 + r_steel3

l_steel1 = occ.Box(occ.Pnt(-1.6-0.5,-15,50+10-3.2),occ.Pnt(-1.6-120.5,-65,50+10))
l_steel2 = occ.Box(occ.Pnt(-1.6-0.5,-15,-(50+10-3.2)),occ.Pnt(-1.6-120.5,-65,-(50+10)))
l_steel3 = occ.Box(occ.Pnt(-1.6-120.5+3.2,-15,50+10-3.2),occ.Pnt(-1.6-120.5,-65,-(50+10)))
l_steel = l_steel1 + l_steel2 + l_steel3

##########################################################################
# Compounding ...
##########################################################################

geom = coil + mid_steel + r_steel + l_steel
ambient =  occ.Box(occ.Pnt(-200,-200,-100), occ.Pnt(200,200,100))
full = occ.Compound([geom, ambient])

##########################################################################
# Identifications
##########################################################################

for face in coil.faces: face.name = 'coil_face'
for face in r_steel.faces: face.name = 'r_steel_face'
for face in l_steel.faces: face.name = 'l_steel_face'
for face in mid_steel.faces: face.name = 'mid_steel_face'
for face in ambient.faces: face.name = 'ambient_face'

coil.mat("coil")
r_steel.mat("r_steel")
l_steel.mat("l_steel")
mid_steel.mat("mid_steel")
    
##########################################################################
# "Fancy" coloring cuz why not I got a bit bored :)
##########################################################################

coil.faces.col=(1,0.5,0)
l_steel.faces.col=(1,0.5,1)
r_steel.faces.col=(1,0.5,1)
mid_steel.faces.col=(1,0.5,1)
ambient.faces.col=(1,1,1)

DrawGeo(full, clipping={"z":-1, "dist":64});

##########################################################################
# Generating mesh...
##########################################################################

geoOCC = occ.OCCGeometry(full)
ng.Draw(geoOCC)

geoOCCmesh = geoOCC.GenerateMesh()
# geoOCCmesh.Refine()
    
##########################################################################
# Extracting info from the mesh
##########################################################################

import numpy as np
import numpy as npy

npoints2D = geoOCCmesh.Elements2D().NumPy()['np'].max()
npoints3D = geoOCCmesh.Elements3D().NumPy()['np'].max()

p = geoOCCmesh.Coordinates()

t = np.c_[geoOCCmesh.Elements3D().NumPy()['nodes'].astype(np.uint64)[:,:npoints3D],
          geoOCCmesh.Elements3D().NumPy()['index'].astype(np.uint64)]-1

f = np.c_[geoOCCmesh.Elements2D().NumPy()['nodes'].astype(np.uint64)[:,:npoints2D],
          geoOCCmesh.Elements2D().NumPy()['index'].astype(np.uint64)]-1

e = np.c_[geoOCCmesh.Elements1D().NumPy()['nodes'].astype(np.uint64)[:,:((npoints2D+1)//2)],
          geoOCCmesh.Elements1D().NumPy()['index'].astype(np.uint64)]-1

max_bc_index = geoOCCmesh.Elements2D().NumPy()['index'].astype(np.uint64).max()
max_rg_index = geoOCCmesh.Elements3D().NumPy()['index'].astype(np.uint64).max()

regions_2d_np = []
for i in range(max_bc_index):
    regions_2d_np += [geoOCCmesh.GetBCName(i)]

regions_3d_np = []
for i in range(max_rg_index):
    regions_3d_np += [geoOCCmesh.GetMaterial(i+1)]

identifications = np.array(geoOCCmesh.GetIdentifications())


all_but_ambient = [i for i, x in enumerate(regions_2d_np) if x != "ambient_face"]


def getIndices2d(liste, name):
    regions = npy.char.split(name,',').tolist()
    ind = npy.empty(shape=(0,),dtype = npy.int64)
    for k in regions:
        if k[0] == '*':
            n = npy.flatnonzero(npy.char.find(liste,k[1:])!=-1)
        else:
            n = npy.flatnonzero(npy.char.equal(liste,k))
        ind = npy.append(ind, n, axis = 0)
    return npy.unique(ind)

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

fig = go.Figure()

ind_regions = getIndices2d(regions_2d_np, 'l_steel_face,r_steel_face,mid_steel_face,coil_face')
indices = npy.in1d(f[:,-1],ind_regions)

f = f[indices,:]

xx_trig = np.c_[p[f[:,0],0],p[f[:,1],0],p[f[:,2],0]]
yy_trig = np.c_[p[f[:,0],1],p[f[:,1],1],p[f[:,2],1]]
zz_trig = np.c_[p[f[:,0],2],p[f[:,1],2],p[f[:,2],2]]

nt = t.shape[0]

# ii, jj, kk = np.r_[:3*nt].reshape((nt, 3)).T
# fig.add_trace(go.Mesh3d(
#     name = 'Trig values',
#     x = xx_trig.flatten(), 
#     y = yy_trig.flatten(), 
#     z = zz_trig.flatten(),
#     i = ii, j = jj, k = kk, intensity = 0*zz_trig.flatten(), 
#     colorscale = 'Rainbow',
#     intensitymode = 'vertex',
#     lighting = dict(ambient = 1),
#     contour_width = 1, contour_color = "#000000", contour_show = True,
#     # xaxis = dict(range())
# ))

xxx_trig = np.c_[xx_trig,xx_trig[:,0],np.nan*xx_trig[:,0]]
yyy_trig = np.c_[yy_trig,yy_trig[:,0],np.nan*yy_trig[:,0]]
zzz_trig = np.c_[zz_trig,zz_trig[:,0],np.nan*zz_trig[:,0]]

fig.add_trace(go.Scatter3d(name = 'Trig traces',
                            mode = 'lines',
                            x = xxx_trig.flatten(),
                            y = yyy_trig.flatten(),
                            z = zzz_trig.flatten(),
                            line = go.scatter3d.Line(color = 'black', 
                                                    width = 1.5),
                            showlegend = False))


# fig.show()

































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

mp_tet = 1/3*(p[t[:,0],:] + p[t[:,1],:] + p[t[:,2],:] + p[t[:,3],:])

e_new = npy.sort(e[:,:2])
f_new = npy.sort(f[:,:3])
    
nt = t.shape[0]

#############################################################################################################
edges = npy.sort(edges_tets).astype(int)
EdgesToVertices, je = npy.unique(edges, axis = 0, return_inverse = True)

NoEdges = EdgesToVertices.shape[0]
TetsToEdges = je[:6*nt].reshape(nt,6, order = 'F').astype(npy.int64)
BoundaryEdges = intersect2d(EdgesToVertices,e_new)
# InteriorEdges = npy.setdiff1d(npy.arange(NoEdges),BoundaryEdges)

EdgesToVertices = npy.c_[EdgesToVertices,npy.zeros(EdgesToVertices.shape[0],dtype = npy.int64)-1]
EdgesToVertices[BoundaryEdges,2] = e[:,-1]
#############################################################################################################



#############################################################################################################
faces = npy.sort(faces_tets).astype(int)
FacesToVertices, je = npy.unique(faces, axis = 0, return_inverse = True)

NoFaces = FacesToVertices.shape[0]
TetsToFaces = je[:4*nt].reshape(nt,4, order = 'F').astype(npy.int64)
BoundaryFaces = intersect2d(FacesToVertices,f_new)
# InteriorFaces = npy.setdiff1d(npy.arange(NoFaces),BoundaryFaces)

FacesToVertices = npy.c_[FacesToVertices,npy.zeros(FacesToVertices.shape[0],dtype = npy.int64)-1]
FacesToVertices[BoundaryFaces,3] = f[:,-1]
#############################################################################################################


if (t.shape[1] == 5): # linear mappings
    t0 = t[:,0]; t1 = t[:,1]; t2 = t[:,2]; t3 = t[:,3]
    C00 = p[t0,0]; C01 = p[t1,0]; C02 = p[t2,0]; C03 = p[t3,0];
    C10 = p[t0,1]; C11 = p[t1,1]; C12 = p[t2,1]; C13 = p[t3,1];
    C20 = p[t0,2]; C21 = p[t1,2]; C22 = p[t2,2]; C23 = p[t3,2];
    
    Fx = lambda x,y,z : C00*(1-x-y-z) + C01*x + C02*y + C03*z
    Fy = lambda x,y,z : C10*(1-x-y-z) + C11*x + C12*y + C13*z
    Fz = lambda x,y,z : C20*(1-x-y-z) + C21*x + C22*y + C23*z
    
    JF00 = lambda x,y,z : C01-C00
    JF01 = lambda x,y,z : C02-C00
    JF02 = lambda x,y,z : C03-C00
    
    JF10 = lambda x,y,z : C11-C10
    JF11 = lambda x,y,z : C12-C10
    JF12 = lambda x,y,z : C13-C10
    
    JF20 = lambda x,y,z : C21-C20
    JF21 = lambda x,y,z : C22-C20
    JF22 = lambda x,y,z : C23-C20
    
    