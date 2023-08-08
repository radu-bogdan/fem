from ngsolve import *
from netgen.geom2d import SplineGeometry
import numpy as np
import netgen.occ as occ

rect = occ.MoveTo(0,-0.02).Rectangle(0.04, 0.02).Face()
ball = occ.MoveTo(0,0).Arc(0.005, 90).Rotate(90).Line(0.005).Close().Face()


domains = []
# domains.append(stator_iron)
# domains.append(rotor_iron)
domains.append(rect)
domains.append(ball)
# domains.append(magnet1)
# domains.append(magnet2)
# domains.append(air_magnet1_1)
# domains.append(air_magnet1_2)

geo = occ.Glue(domains)

geoOCC = occ.OCCGeometry(geo, dim=2)
geoOCCmesh = geoOCC.GenerateMesh()

import sys
sys.path.insert(0,'../../') # adds parent directory

import pde
import ngsolve as ng

meshng = ng.Mesh(geoOCCmesh)
meshng.Refine()
meshng.Refine()
MESH = pde.mesh.netgen(meshng.ngmesh)


# geoOCCmesh.Refine()
# geoOCCmesh.Refine()
# geoOCCmesh.Refine()
MESH.pdemesh2()