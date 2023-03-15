# -----------------------------------------------------------------------------
#
#  Gmsh Python extended tutorial 5
#
#  Additional geometrical data: parametrizations, normals, curvatures
#
# -----------------------------------------------------------------------------

import gmsh
import sys
import math

gmsh.initialize(sys.argv)

# The API provides access to geometrical data in a CAD kernel agnostic manner.

# Let's create a simple CAD model by fusing a sphere and a cube, then mesh the
# surfaces:
    
    
# gmsh.model.occ.addRectangle(0, 0, 0, 1, 2)
# gmsh.model.occ.addDisk(0,0,0,1,1)
# gmsh.model.occ.cut([(2, 1)], [(2, 2)])
# gmsh.model.occ.addDisk(0,0,0,1,1)
# gmsh.model.occ.fragment([(3, 1)], [(2, 7)])

# gmsh.model.occ.healShapes()

# gmsh.model.occ.synchronize()

# gmsh.model.mesh.generate(2)
gmsh.fltk.run()

gmsh.finalize()
