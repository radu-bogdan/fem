import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

import meshio
import numpy as np
import pde


import plotly.io as pio
pio.renderers.default = 'browser'

mesh = meshio.read(
    "Motor_Bosch_2d_new.vol",  # string, os.PathLike, or a buffer/open file
    file_format = "netgen",  # optional if filename is a path; inferred from extension
    # see meshio-convert -h for all possible formats
)

# mesh.write(
#     "foo.msh",  # str, os.PathLike, or buffer/open file
#     file_format = "gmsh",  # optional if first argument is a path; inferred from extension
# )

p = mesh.points
t = np.c_[mesh.cells_dict['triangle'],
          mesh.cell_data_dict['netgen:index']['triangle']]
e = np.c_[mesh.cells_dict['line'],
          mesh.cell_data_dict['netgen:index']['line']]
q = np.empty(0)

MESH = pde.mesh(p,e,t,q)

vec = mesh.cell_data_dict['netgen:index']['triangle']


fig = MESH.pdemesh()
# fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 1), vec)
fig.show()