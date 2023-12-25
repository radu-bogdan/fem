from t13geo import *

import sys
sys.path.insert(0,'../../') # adds parent directory
import pde
from sksparse.cholmod import cholesky as chol

order = 1

MESH = pde.mesh3(p,e,f,t,regions_3d_np,regions_2d_np)

phi_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'M', order = order)
dphix_H1, dphiy_H1, dphiz_H1 = pde.h1.assemble3(MESH, space = 'P1', matrix = 'K', order = order)
D = pde.int.assemble3(MESH, order = order)

M = phi_H1 @ D @ phi_H1.T

K = dphix_H1 @ D @ dphix_H1.T +\
    dphiy_H1 @ D @ dphiy_H1.T +\
    dphiz_H1 @ D @ dphiz_H1.T

R0, RSS = pde.h1.assembleR3(MESH, space = 'P1', faces = 'ambient_face')

Kn = RSS @ K @ RSS.T

coeff = lambda x,y,z : 1+0*x*y*z
J = pde.int.evaluate3(MESH, order = order, coeff = coeff, regions = 'mid_steel').diagonal()

r = J @ D @ phi_H1.T

# # solve:
u = RSS.T@(chol(Kn).solve_A(RSS@r))
# u = chol(M).solve_A(r)

u2 = coeff(p[:,0],p[:,1],p[:,2])

##########################################################################
# Plotting fun
##########################################################################

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

fig = go.Figure()


all_but_ambient = [i for i, x in enumerate(regions_2d_np) if x != 'ambient_face']
ind_regions = MESH.getIndices2d(regions_2d_np, 'l_steel_face,r_steel_face,mid_steel_face,coil_face')
indices = npy.in1d(f[:,-1],ind_regions)


# fn = f.copy()
f = f[indices,:3]

xx_trig = np.c_[p[f[:,0],0],p[f[:,1],0],p[f[:,2],0]]
yy_trig = np.c_[p[f[:,0],1],p[f[:,1],1],p[f[:,2],1]]
zz_trig = np.c_[p[f[:,0],2],p[f[:,1],2],p[f[:,2],2]]

nt = t.shape[0]
nf = f.shape[0]


ii, jj, kk = np.r_[:3*nf].reshape((nf, 3)).T
fig.add_trace(go.Mesh3d(
    name = 'Trig values',
    x = xx_trig.flatten(), 
    y = yy_trig.flatten(), 
    z = zz_trig.flatten(),
    i = ii, j = jj, k = kk, intensity = u[f].flatten(),
    colorscale = 'Rainbow',
    intensitymode = 'vertex',
    lighting = dict(ambient = 1),
    contour_width = 1, contour_color = "#000000", contour_show = True,
    # xaxis = dict(range())
))

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



fig.show()

# f = fn.copy()
##########################################################################
