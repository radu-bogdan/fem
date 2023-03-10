#!/usr/bin/python --relpath_append ../

import sys
sys.path.insert(0,'../../') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import geometries

import plotly.io as pio
pio.renderers.default = 'browser'
from matplotlib.pyplot import spy
# pio.renderers.default = 'svg'


np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = np.inf)
np.set_printoptions(precision = 8)

# p,e,t,q = pde.petq_from_gmsh(filename = 'unit_square.geo',hmax = 0.3)

# gmsh.initialize()

d = 3
l = 10
gmsh.initialize()
gmsh.model.add("Capacitor plates")
geometries.capacitorPlates(a = 20,b = 20,c = 0.5,d = d,l = l)
gmsh.option.setNumber("Mesh.Algorithm", 2)
gmsh.option.setNumber("Mesh.MeshSizeMax", 2)
# gmsh.fltk.run()
p,e,t,q = pde.petq_generate()

MESH = pde.mesh(p,e,t,q)

# TODO:  MESH = pde.refinemesh(p,e,t,q)
BASIS = pde.basis()
LISTS = pde.lists(MESH)

f = lambda x,y : 0*np.sin(np.pi*x)*np.sin(np.pi*y)
g1 = lambda x,y : -1+0*x
g2 = lambda x,y :  1+0*x

# TODO : iwas stimmt net wenn ma quads hat
Kx,Ky,KL,KL = pde.assemble.h1(MESH,BASIS,LISTS,dict(space = 'P2', matrix = 'K'))
M = pde.assemble.h1(MESH,BASIS,LISTS,dict(space = 'P2', matrix = 'M'))

sizeM = M.shape[0]

walls = np.r_[5,6,7,8,9,10,11,12]
left_block = np.r_[5,6,7,8]
right_block = np.r_[9,10,11,12]

B_full = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P2', size = sizeM))
B_left_block = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P2', size = sizeM, edges = left_block))
B_right_block = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P2', size = sizeM, edges = right_block))
B_walls = pde.assemble.h1b(MESH,BASIS,LISTS,dict(space = 'P2', size = sizeM, edges = walls))
M_f = pde.projections.assemH1(MESH, BASIS, LISTS, dict(trig = 'P2'), f)

B_g  = pde.projections.assem_H1_b(MESH, BASIS, LISTS, dict(space = 'P2', order = 2, edges = left_block, size = sizeM), g1)
B_g += pde.projections.assem_H1_b(MESH, BASIS, LISTS, dict(space = 'P2', order = 2, edges = right_block, size = sizeM), g2)

# M = MAT['P1-Q1']['M']
# Kx = MAT['P1-Q1']['Kx']
# Ky = MAT['P1-Q1']['Ky']

gamma = 10**8

A = Kx + Ky + gamma*B_walls
b = gamma*B_g + M_f



tm = time.time()
phi = sps.linalg.spsolve(A,b)

oneM = np.ones(sizeM)
Q = gamma*(oneM@B_left_block@(phi+1)) #+ oneM@B_right_block@(phi-1)
# Q = gamma*(oneM@B_right_block@(phi-1))

print(str(Q/2), ' soll: ', str(l/d))

elapsed = time.time()-tm
print('Solving took ' + str(elapsed)[0:5] + ' seconds.')

fig = MESH.pdesurf_hybrid(dict(trig = 'P1', controls = 1), phi[0:MESH.np])
fig.show()


# def show_in_window(fig):
#     import sys, os
#     import plotly.offline
#     from PyQt5.QtCore import QUrl
#     from PyQt5.QtWebEngineWidgets import QWebEngineView
#     from PyQt5.QtWidgets import QApplication
    
#     plotly.offline.plot(fig, filename='name.html', auto_open=False)
    
#     app = QApplication(sys.argv)
#     web = QWebEngineView()
#     file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "name.html"))
#     web.load(QUrl.fromLocalFile(file_path))
#     web.show()
#     sys.exit(app.exec_())


# show_in_window(fig)

# from matplotlib.pyplot import spy
# spy(Kx+Ky,markersize=0.05)