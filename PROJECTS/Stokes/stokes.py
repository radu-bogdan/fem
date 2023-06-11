import sys
sys.path.insert(0,'../../') # adds parent directory
import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
# from sksparse.cholmod import cholesky as chol
import plotly.io as pio
pio.renderers.default = 'browser'
import numba as nb
import pyamg

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
cmap = plt.cm.jet
from sparse_dot_mkl import dot_product_mkl as mult


init_ref = 0.05

################################################################################
gmsh.initialize()
gmsh.open('twoDomains.geo')
gmsh.option.setNumber("Mesh.Algorithm", 1)
gmsh.option.setNumber("Mesh.MeshSizeMax", init_ref)
gmsh.option.setNumber("Mesh.MeshSizeMin", init_ref)
gmsh.option.setNumber("Mesh.SaveAll", 1)

p,e,t,q = pde.petq_generate()
MESH = pde.mesh(p,e,t,q)
gmsh.finalize()
################################################################################

MESH.pdemesh2()


dphix_H1, dphiy_H1 = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 1)











