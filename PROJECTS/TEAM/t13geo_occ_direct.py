from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox,BRepPrimAPI_MakeCylinder
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir


box1 = BRepPrimAPI_MakeBox(gp_Pnt(-100, -100, -50), gp_Pnt(100, 100, 50)).Shape()
box2 = BRepPrimAPI_MakeBox(gp_Pnt(-75,-75,-50), gp_Pnt(75,75,50)).Shape()

##########################################################################
# Rounding corners ...
##########################################################################

corner1_ext = BRepPrimAPI_MakeBox(gp_Pnt(75,75,-50), gp_Pnt(100,100,50))
cyl1_ext = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(75,75,-50), gp_Dir(0,0,1)), 25, 100)
corner1_int = BRepPrimAPI_MakeBox(gp_Pnt(50,50,-50), gp_Pnt(75,75,50))
cyl1_int = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(50,50,-50), gp_Dir(0,0,1)), 25, 100)

corner1_int = corner1_int-cyl1_int; corner1_ext = corner1_ext-cyl1_ext

# corner1_ext = occ.Box(occ.Pnt(75,75,-50), occ.Pnt(100,100,50))
# cyl1_ext = occ.Cylinder(occ.Pnt(75,75,-50), occ.Z, r=25, h=100)
# corner1_int = occ.Box(occ.Pnt(50,50,-50), occ.Pnt(75,75,50))
# cyl1_int = occ.Cylinder(occ.Pnt(50,50,-50), occ.Z, r=25, h=100)
# corner1_int = corner1_int-cyl1_int; corner1_ext = corner1_ext-cyl1_ext


stop

from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.gp import gp_Ax2, gp_Dir



from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE

# Function to iterate over the edges of a shape
def get_edges(shape):
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        yield explorer.Current()
        explorer.Next()

# Function to round corners of a box, using the get_edges function
def round_box_corners(box, radius):
    fillet_maker = BRepFilletAPI_MakeFillet(box)
    for edge in get_edges(box):
        fillet_maker.Add(radius, edge)

    return fillet_maker.Shape()


# Function to create a cylinder
def create_cylinder(base_point, direction, radius, height):
    axis = gp_Ax2(gp_Pnt(*base_point), gp_Dir(*direction))
    cylinder = BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()

    return cylinder

# Apply rounding to the boxes (example radius)
rounded_box1 = round_box_corners(box1, 5)
rounded_box2 = round_box_corners(box2, 5)

# Create a cylinder (example parameters)
cylinder = create_cylinder((0, 0, 0), (0, 0, 1), 10, 50)














