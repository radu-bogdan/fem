from netgen.webgui import Draw as DrawGeo
import netgen.occ as occ
import ngsolve as ng
from netgen.meshing import IdentificationType
import numpy as np
import netgen as netg

from netgen.webgui import Draw as DrawGeo
from ngsolve.webgui import Draw as DrawMesh

a = occ.Rectangle(1,1).Face()

# a.Rotation(occ.Axis((0,0,0), occ.X), 90)

a.edges.Max(occ.X).name = "right"
a.edges.Min(occ.X).name = "left"
a.edges.Max(occ.Y).name = "top"
a.edges.Min(occ.Y).name = "bot"

# rot2 = occ.MoveTo(1,0)
rot = occ.Rotation(occ.Axis((0,0,0), occ.Z), 90)
a.edges.Min(occ.Y).Identify(a.edges.Min(occ.X),"per",IdentificationType.PERIODIC,rot)


# a.edges.Max(occ.Y).Identify(a.edges.Min(occ.Y), "br")
# a.edges.Max(occ.X).Identify(a.edges.Min(occ.X), "lr")

geo = occ.OCCGeometry(a,dim = 2)
mesh = ng.Mesh(geo.GenerateMesh(maxh=0.1))

plist = []
for pair in mesh.ngmesh.GetIdentifications():
    plist += list(mesh.vertices[pair[0]-1].point) + [0]
    plist += list(mesh.vertices[pair[1]-1].point) + [0]
    
DrawMesh(mesh, objects=[{"type" : "lines", 
                         "position" : plist, 
                         "name": "identification", 
                         "color": "purple"}]);

# print(rot((0,0,1)))
print(mesh.ngmesh.GetIdentifications())
# dir(rot)

# dir(occ)