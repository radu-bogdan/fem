from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt

def create_box(corner1, corner2):
    pnt1 = gp_Pnt(corner1[0], corner1[1], corner1[2])
    pnt2 = gp_Pnt(corner2[0], corner2[1], corner2[2])
    dx = abs(corner2[0] - corner1[0])
    dy = abs(corner2[1] - corner1[1])
    dz = abs(corner2[2] - corner1[2])

    # The BRepPrimAPI_MakeBox function can also take dimensions directly
    box = BRepPrimAPI_MakeBox(pnt1, dx, dy, dz).Shape()

    return box

# Creating the boxes similar to the original example
box1 = create_box((-100, -100, -50), (100, 100, 50))
box2 = create_box((-75, -75, -50), (75, 75, 50))

# These boxes are now created as TopoDS_Shape objects and can be used for further operations.




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














