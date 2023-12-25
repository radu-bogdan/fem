import ngsolve as ng
import netgen.occ as occ
# import netgen.gui
from netgen.webgui import Draw as DrawGeo

full = occ.Box(occ.Pnt(-100,-100,-50), occ.Pnt(100,100,50))

##########################################################################
# Identifications
##########################################################################

for face in full.faces: face.name = 'ambient_face'

full.mat("full")

##########################################################################
# Generating mesh...
##########################################################################

geoOCC = occ.OCCGeometry(full)
ng.Draw(geoOCC)

geoOCCmesh = geoOCC.GenerateMesh(maxh=10)
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


##########################################################################
# Plotting fun
##########################################################################

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


fn = f.copy()
f = f[indices,:]

xx_trig = np.c_[p[f[:,0],0],p[f[:,1],0],p[f[:,2],0]]
yy_trig = np.c_[p[f[:,0],1],p[f[:,1],1],p[f[:,2],1]]
zz_trig = np.c_[p[f[:,0],2],p[f[:,1],2],p[f[:,2],2]]

nt = t.shape[0]

f = fn.copy()

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

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
from sksparse.cholmod import cholesky as chol

MESH = pde.mesh3(p,e,f,t,regions_3d_np,regions_2d_np)

phi_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = 4)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = 4)
D = pde.int.assemble3(MESH, order = 4)

M = phi_H1 @ D @ phi_H1.T

K = dphix_H1 @ D @ dphix_H1.T +\
    dphiy_H1 @ D @ dphiy_H1.T +\
    dphiz_H1 @ D @ dphiz_H1.T

R0, RSS = pde.h1.assembleR3(MESH, space = 'P1', faces = 'ambient_face')

Kn = RSS @ K @ RSS.T







    