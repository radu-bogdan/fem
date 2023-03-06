import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

import gmsh
import geometries
import pde

gmsh.initialize()
gmsh.model.add("Capacitor plates")
# geometries.unitSquare()
gmsh.open('zeger.geo_unrolled')
gmsh.option.setNumber("Mesh.Algorithm", 1)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 0.001)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
gmsh.option.setNumber("Mesh.SaveAll", 1)
p,e,t,q = pde.petq_generate()
gmsh.fltk.run()
gmsh.finalize()

MESH = pde.mesh(p,e,t,q)

fig = MESH.pdemesh()
fig.show()