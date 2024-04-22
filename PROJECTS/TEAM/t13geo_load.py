print('t13_geo')

# import ngsolve as ng
# import netgen.occ as occ
# import time

from imports import *

geoOCCmesh = ng.Mesh('whatever2.vol').ngmesh

MESH = pde.mesh3.netgen(geoOCCmesh)

# geoOCCmesh.SecondOrder()
# geoOCCmesh.Refine()
# geoOCCmesh.Refine()
# geoOCCmesh.Refine()

# print('Generating the mesh took ...', time.monotonic()-tm)