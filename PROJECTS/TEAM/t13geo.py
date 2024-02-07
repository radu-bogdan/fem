import ngsolve as ng
import netgen.occ as occ

# import netgen.gui
# from netgen.webgui import Draw as DrawGeo

box1 = occ.Box(occ.Pnt(-100,-100,-50), occ.Pnt(100,100,50))
box2 = occ.Box(occ.Pnt(-75,-75,-50), occ.Pnt(75,75,50))
    
##########################################################################
# Rounding corners ...
##########################################################################

corner1_ext = occ.Box(occ.Pnt(75,75,-50), occ.Pnt(100,100,50))
cyl1_ext = occ.Cylinder(occ.Pnt(75,75,-50), occ.Z, r=25, h=100)
corner1_int = occ.Box(occ.Pnt(50,50,-50), occ.Pnt(75,75,50))
cyl1_int = occ.Cylinder(occ.Pnt(50,50,-50), occ.Z, r=25, h=100)
corner1_int = corner1_int-cyl1_int; corner1_ext = corner1_ext-cyl1_ext

corner2_ext = occ.Box(occ.Pnt(-100,-100,-50), occ.Pnt(-75,-75,50))
cyl2_ext = occ.Cylinder(occ.Pnt(-75,-75,-50), occ.Z, r=25, h=100)
corner2_int = occ.Box(occ.Pnt(-75,-75,-50), occ.Pnt(-50,-50,50))
cyl2_int = occ.Cylinder(occ.Pnt(-50,-50,-50), occ.Z, r=25, h=100)
corner2_int = corner2_int-cyl2_int; corner2_ext = corner2_ext-cyl2_ext

corner3_ext = occ.Box(occ.Pnt(75,-75,-50), occ.Pnt(100,-100,50))
cyl3_ext = occ.Cylinder(occ.Pnt(75,-75,-50), occ.Z, r=25, h=100)
corner3_int = occ.Box(occ.Pnt(50,-50,-50), occ.Pnt(75,-75,50))
cyl3_int = occ.Cylinder(occ.Pnt(50,-50,-50), occ.Z, r=25, h=100)
corner3_int = corner3_int-cyl3_int; corner3_ext = corner3_ext-cyl3_ext

corner4_ext = occ.Box(occ.Pnt(-75,75,-50), occ.Pnt(-100,100,50))
cyl4_ext = occ.Cylinder(occ.Pnt(-75,75,-50), occ.Z, r=25, h=100)
corner4_int = occ.Box(occ.Pnt(-50,50,-50), occ.Pnt(-75,75,50))
cyl4_int = occ.Cylinder(occ.Pnt(-50,50,-50), occ.Z, r=25, h=100)
corner4_int = corner4_int-cyl4_int; corner4_ext = corner4_ext-cyl4_ext

##########################################################################
# Adding the steel parts
##########################################################################

coil_full = (box1-box2)+corner1_int-corner1_ext+corner2_int-corner2_ext+corner3_int-corner3_ext+corner4_int-corner4_ext

mid_steel = occ.Box(occ.Pnt(-1.6,-25,-64.2),occ.Pnt(1.6,25,64.2))

r_steel1 = occ.Box(occ.Pnt(1.6+0.5,15,50+10-3.2),occ.Pnt(1.6+120.5,65,50+10))
r_steel2 = occ.Box(occ.Pnt(1.6+0.5,15,-(50+10-3.2)),occ.Pnt(1.6+120.5,65,-(50+10)))
r_steel3 = occ.Box(occ.Pnt(1.6+120.5-3.2,15,50+10-3.2),occ.Pnt(1.6+120.5,65,-(50+10)))
r_steel = r_steel1 + r_steel2 + r_steel3

l_steel1 = occ.Box(occ.Pnt(-1.6-0.5,-15,50+10-3.2),occ.Pnt(-1.6-120.5,-65,50+10))
l_steel2 = occ.Box(occ.Pnt(-1.6-0.5,-15,-(50+10-3.2)),occ.Pnt(-1.6-120.5,-65,-(50+10)))
l_steel3 = occ.Box(occ.Pnt(-1.6-120.5+3.2,-15,50+10-3.2),occ.Pnt(-1.6-120.5,-65,-(50+10)))
l_steel = l_steel1 + l_steel2 + l_steel3

##########################################################################
# Glueing ...
##########################################################################

half_box_1 = occ.Box(occ.Pnt(-100,-100,-50), occ.Pnt(0,100,50))
half_box_2 = occ.Box(occ.Pnt(100,-100,-50), occ.Pnt(0,100,50))

coil_half_box_1 = coil_full*half_box_1
coil_half_box_2 = coil_full*half_box_2

coil = occ.Glue([coil_half_box_1,coil_half_box_2])
ambient =  occ.Box(occ.Pnt(-200,-200,-100), occ.Pnt(200,200,100))

full = occ.Glue([coil, mid_steel, r_steel, l_steel, ambient])


##########################################################################
# Identifications
##########################################################################


for face in coil.faces: face.name = 'coil_face'
for face in r_steel.faces: face.name = 'r_steel_face'
for face in l_steel.faces: face.name = 'l_steel_face'
for face in mid_steel.faces: face.name = 'mid_steel_face'
for face in ambient.faces: face.name = 'ambient_face'


steel_h = 2.5

for face in r_steel.faces: face.maxh = steel_h
for face in l_steel.faces: face.maxh = steel_h
for face in mid_steel.faces: face.maxh = steel_h


coil.faces[6].name = 'coil_cut_1'
coil.faces[12].name = 'coil_cut_2'

coil.mat("coil")
r_steel.mat("r_steel")
l_steel.mat("l_steel")
mid_steel.mat("mid_steel")
ambient.mat("ambient")

##########################################################################
# Generating mesh...
##########################################################################

geoOCC = occ.OCCGeometry(full)
# ng.Draw(geoOCC)

geoOCCmesh = geoOCC.GenerateMesh()
# geoOCCmesh.Refine()
# geoOCCmesh.Refine()