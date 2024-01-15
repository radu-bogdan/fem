from netgen.occ import *
from ngsolve import *
import math

cyl = Cylinder((0,0,0), Z, r=0.01, h=0.03).faces[0]
heli = Edge(Segment((0,0), (12*math.pi, 0.03)), cyl)
ps = heli.start
vs = heli.start_tangent
pe = heli.end
ve = heli.end_tangent

e1 = Segment((0,0,-0.03), (0,0,-0.01))
c1 = BezierCurve( [(0,0,-0.01), (0,0,0), ps-vs, ps])
e2 = Segment((0,0,0.04), (0,0,0.06))
c2 = BezierCurve( [pe, pe+ve, (0,0,0.03), (0,0,0.04)])
spiral = Wire([e1, c1, heli, c2, e2])
circ = Face(Wire([Circle((0,0,-0.03), Z, 0.001)]))
coil = Pipe(spiral, circ)

coil.faces.maxh=0.2
coil.faces.name="coilbnd"
coil.faces.Max(Z).name="in"
coil.faces.Min(Z).name="out"
coil.mat("coil")
crosssection = coil.faces.Max(Z).mass



box = Box((-0.04,-0.04,-0.03), (0.04,0.04,0.06))
box.faces.name = "outer"
air = box-coil
air.mat("air");


geo = OCCGeometry(Glue([coil,air]))
with TaskManager():
    geoOCCmesh = geo.GenerateMesh(meshsize.coarse, maxh=1)
print('finished erzeugen')