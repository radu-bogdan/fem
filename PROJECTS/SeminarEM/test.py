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
    
    
gmsh.model.occ.addRectangle(0, 0, 0, 1, 2)
gmsh.model.occ.addCircle(0,0,0,1)
# gmsh.model.occ.fuse([(2, sq)], [(2, rotor_inner)])



gmsh.model.occ.synchronize()

gmsh.model.mesh.generate(2)
gmsh.fltk.run()

gmsh.finalize()
