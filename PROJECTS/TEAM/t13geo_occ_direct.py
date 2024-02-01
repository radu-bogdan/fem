from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox,BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse,BRepAlgoAPI_Common,BRepAlgoAPI_Cut
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCC.Core.TopAbs import TopAbs_SHAPE



box1 = BRepPrimAPI_MakeBox(gp_Pnt(-100, -100, -50), gp_Pnt(100, 100, 50)).Shape()
box2 = BRepPrimAPI_MakeBox(gp_Pnt(-75,-75,-50), gp_Pnt(75,75,50)).Shape()

##########################################################################
# Rounding corners ...
##########################################################################

corner1_ext = BRepPrimAPI_MakeBox(gp_Pnt(75,75,-50), gp_Pnt(100,100,50)).Shape()
cyl1_ext = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(75,75,-50), gp_Dir(0,0,1)), 25, 100).Shape()
corner1_int = BRepPrimAPI_MakeBox(gp_Pnt(50,50,-50), gp_Pnt(75,75,50)).Shape()
cyl1_int = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(50,50,-50), gp_Dir(0,0,1)), 25, 100).Shape()
corner1_int = BRepAlgoAPI_Cut(corner1_int, cyl1_int).Shape(); corner1_ext = BRepAlgoAPI_Cut(corner1_ext,cyl1_ext).Shape()

corner2_ext = BRepPrimAPI_MakeBox(gp_Pnt(-100,-100,-50), gp_Pnt(-75,-75,50)).Shape()
cyl2_ext = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-75,-75,-50), gp_Dir(0,0,1)), 25, 100).Shape()
corner2_int = BRepPrimAPI_MakeBox(gp_Pnt(-75,-75,-50), gp_Pnt(-50,-50,50)).Shape()
cyl2_int = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-50,-50,-50), gp_Dir(0,0,1)), 25, 100).Shape()
corner2_int = BRepAlgoAPI_Cut(corner2_int, cyl2_int).Shape(); corner2_ext = BRepAlgoAPI_Cut(corner2_ext,cyl2_ext).Shape()

corner3_ext = BRepPrimAPI_MakeBox(gp_Pnt(75,-75,-50), gp_Pnt(100,-100,50)).Shape()
cyl3_ext = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(75,-75,-50), gp_Dir(0,0,1)), 25, 100).Shape()
corner3_int = BRepPrimAPI_MakeBox(gp_Pnt(50,-50,-50), gp_Pnt(75,-75,50)).Shape()
cyl3_int = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(50,-50,-50), gp_Dir(0,0,1)), 25, 100).Shape()
corner3_int = BRepAlgoAPI_Cut(corner3_int, cyl3_int).Shape(); corner3_ext = BRepAlgoAPI_Cut(corner3_ext,cyl3_ext).Shape()

corner4_ext = BRepPrimAPI_MakeBox(gp_Pnt(-75,75,-50), gp_Pnt(-100,100,50)).Shape()
cyl4_ext = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-75,75,-50), gp_Dir(0,0,1)), 25, 100).Shape()
corner4_int = BRepPrimAPI_MakeBox(gp_Pnt(-50,50,-50), gp_Pnt(-75,75,50)).Shape()
cyl4_int = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-50,50,-50), gp_Dir(0,0,1)), 25, 100).Shape()
corner4_int = BRepAlgoAPI_Cut(corner4_int, cyl4_int).Shape(); corner4_ext = BRepAlgoAPI_Cut(corner4_ext,cyl4_ext).Shape()

##########################################################################
# Adding the steel parts
##########################################################################

def fuse_multiple_shapes(shapes):
    if not shapes:
        raise ValueError("No shapes provided for fusion.")

    # Start with the first shape as the initial result
    fused_shape = shapes[0]

    # Sequentially fuse each shape with the current result
    for shape in shapes[1:]:
        fuse = BRepAlgoAPI_Fuse(fused_shape, shape)
        fused_shape = fuse.Shape()  # Update the result with the new fusion

    return fused_shape



coil_full = fuse_multiple_shapes([BRepAlgoAPI_Cut(box1,box2).Shape(),
            BRepAlgoAPI_Cut(corner1_int,corner1_ext).Shape(),
            BRepAlgoAPI_Cut(corner2_int,corner2_ext).Shape(),
            BRepAlgoAPI_Cut(corner3_int,corner3_ext).Shape(),
            BRepAlgoAPI_Cut(corner4_int,corner4_ext).Shape()])

mid_steel = BRepPrimAPI_MakeBox(gp_Pnt(-1.6,-25,-64.2),gp_Pnt(1.6,25,64.2)).Shape()

r_steel1 = BRepPrimAPI_MakeBox(gp_Pnt(1.6+0.5,15,50+10-3.2),gp_Pnt(1.6+120.5,65,50+10)).Shape()
r_steel2 = BRepPrimAPI_MakeBox(gp_Pnt(1.6+0.5,15,-(50+10-3.2)),gp_Pnt(1.6+120.5,65,-(50+10))).Shape()
r_steel3 = BRepPrimAPI_MakeBox(gp_Pnt(1.6+120.5-3.2,15,50+10-3.2),gp_Pnt(1.6+120.5,65,-(50+10))).Shape()
r_steel = fuse_multiple_shapes([r_steel1,r_steel2,r_steel3])

l_steel1 = BRepPrimAPI_MakeBox(gp_Pnt(-1.6-0.5,-15,50+10-3.2),gp_Pnt(-1.6-120.5,-65,50+10)).Shape()
l_steel2 = BRepPrimAPI_MakeBox(gp_Pnt(-1.6-0.5,-15,-(50+10-3.2)),gp_Pnt(-1.6-120.5,-65,-(50+10))).Shape()
l_steel3 = BRepPrimAPI_MakeBox(gp_Pnt(-1.6-120.5+3.2,-15,50+10-3.2),gp_Pnt(-1.6-120.5,-65,-(50+10))).Shape()
l_steel = fuse_multiple_shapes([l_steel1,l_steel2,l_steel3])

##########################################################################
# Glueing ...
##########################################################################

half_box_1 = BRepPrimAPI_MakeBox(gp_Pnt(-100,-100,-50), gp_Pnt(0,100,50)).Shape()
half_box_2 = BRepPrimAPI_MakeBox(gp_Pnt(100,-100,-50), gp_Pnt(0,100,50)).Shape()

coil_half_box_1 = BRepAlgoAPI_Common(coil_full,half_box_1)
coil_half_box_2 = BRepAlgoAPI_Common(coil_full,half_box_2)

def glue_shapes(shapes):
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    
    # Add each shape to the compound, ensuring it's a valid TopoDS_Shape
    for shape in shapes:
        if isinstance(shape, TopoDS_Shape):
            builder.Add(compound, shape)
        else:
            print("One of the provided shapes is not a TopoDS_Shape or derived from it.")

    return compound

# TODO : evtl finishen... aber warum... ?

stop

coil = glue_shapes([coil_half_box_1,coil_half_box_2])
ambient =  occ.Box(occ.Pnt(-200,-200,-100), occ.Pnt(200,200,100))

full = occ.Glue([coil, mid_steel, r_steel, l_steel, ambient])







##########################################################################


from OCC.Display.SimpleGui import init_display
def display_shapes(*shapes):
    display, start_display, add_menu, add_function_to_menu = init_display()
    for shape in shapes:
        display.DisplayShape(shape, update=True)
    start_display()

display_shapes(coil_full)

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














