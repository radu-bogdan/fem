print('t13_geo')

# import ngsolve as ng
# import netgen.occ as occ
# import time

from imports import *

# import netgen.gui
# from netgen.webgui import Draw as DrawGeo

tm = time.monotonic()
println('Generating the geometry took ...  ')

box1 = occ.Box(occ.Pnt(-0.100,-0.100,-0.050), occ.Pnt(0.100,0.100,0.050))
box2 = occ.Box(occ.Pnt(-0.075,-0.075,-0.050), occ.Pnt(0.075,0.075,0.050))
    
##########################################################################
# Rounding corners ...
##########################################################################

corner1_ext = occ.Box(occ.Pnt(0.075,0.075,-0.050), occ.Pnt(0.100,0.100,0.050))
cyl1_ext = occ.Cylinder(occ.Pnt(0.075,0.075,-0.050), occ.Z, r=0.025, h=0.100)
corner1_int = occ.Box(occ.Pnt(0.050,0.050,-0.050), occ.Pnt(0.075,0.075,0.050))
cyl1_int = occ.Cylinder(occ.Pnt(0.050,0.050,-0.050), occ.Z, r=0.025, h=0.100)
corner1_int = corner1_int-cyl1_int; corner1_ext = corner1_ext-cyl1_ext

corner2_ext = occ.Box(occ.Pnt(-0.100,-0.100,-0.050), occ.Pnt(-0.075,-0.075,0.050))
cyl2_ext = occ.Cylinder(occ.Pnt(-0.075,-0.075,-0.050), occ.Z, r=0.025, h=0.100)
corner2_int = occ.Box(occ.Pnt(-0.075,-0.075,-0.050), occ.Pnt(-0.050,-0.050,0.050))
cyl2_int = occ.Cylinder(occ.Pnt(-0.050,-0.050,-0.050), occ.Z, r=0.025, h=0.100)
corner2_int = corner2_int-cyl2_int; corner2_ext = corner2_ext-cyl2_ext

corner3_ext = occ.Box(occ.Pnt(0.075,-0.075,-0.050), occ.Pnt(0.100,-0.100,0.050))
cyl3_ext = occ.Cylinder(occ.Pnt(0.075,-0.075,-0.050), occ.Z, r=0.025, h=0.100)
corner3_int = occ.Box(occ.Pnt(0.050,-0.050,-0.050), occ.Pnt(0.075,-0.075,0.050))
cyl3_int = occ.Cylinder(occ.Pnt(0.050,-0.050,-0.050), occ.Z, r=0.025, h=0.100)
corner3_int = corner3_int-cyl3_int; corner3_ext = corner3_ext-cyl3_ext

corner4_ext = occ.Box(occ.Pnt(-0.075,0.075,-0.050), occ.Pnt(-0.100,0.100,0.050))
cyl4_ext = occ.Cylinder(occ.Pnt(-0.075,0.075,-0.050), occ.Z, r=0.025, h=0.100)
corner4_int = occ.Box(occ.Pnt(-0.050,0.050,-0.050), occ.Pnt(-0.075,0.075,0.050))
cyl4_int = occ.Cylinder(occ.Pnt(-0.050,0.050,-0.050), occ.Z, r=0.025, h=0.100)
corner4_int = corner4_int-cyl4_int; corner4_ext = corner4_ext-cyl4_ext

coil_full = (box1-box2)+corner1_int-corner1_ext+corner2_int-corner2_ext+corner3_int-corner3_ext+corner4_int-corner4_ext

##########################################################################
# Adding the steel parts
##########################################################################

mid_steel = occ.Sphere(occ.Pnt(0,0,0.05),0.05)

##########################################################################
# Glueing ...
##########################################################################

half_box_1 = occ.Box(occ.Pnt(-0.100,-0.100,-0.050), occ.Pnt(0,0.100,0.050))
half_box_2 = occ.Box(occ.Pnt(0.100,-0.100,-0.050), occ.Pnt(0,0.100,0.050))

coil_half_box_1 = coil_full*half_box_1
coil_half_box_2 = coil_full*half_box_2

coil = occ.Glue([coil_half_box_1,coil_half_box_2])
ambient =  occ.Box(occ.Pnt(-0.200,-0.200,-0.200), occ.Pnt(0.200,0.200,0.200))

full = occ.Glue([coil, mid_steel, ambient])
# full = occ.Glue([coil, mid_steel, r_steel, l_steel, ambient])

##########################################################################
# Identifications
##########################################################################


for face in coil.faces: face.name = 'coil_face'
# for face in r_steel.faces: face.name = 'r_steel_face'
# for face in l_steel.faces: face.name = 'l_steel_face'
# for face in mid_steel.faces: face.name = 'mid_steel_face'
for face in ambient.faces: face.name = 'ambient_face'

coil.faces[6].name = 'coil_cut_1'
coil.faces[12].name = 'coil_cut_2'

coil.mat("coil")
# r_steel.mat("r_steel")
# l_steel.mat("l_steel")
mid_steel.mat("mid_steel")

# for face in mid_steel.faces: face.maxh = 0.002
for face in mid_steel.faces: face.maxh = 0.004

geoOCC = occ.OCCGeometry(full)
geoOCCmesh = geoOCC.GenerateMesh()

MESH = pde.mesh3.netgen(geoOCCmesh)
print(MESH)

print('Generating the mesh took ...', time.monotonic()-tm)

# geoOCCmesh.SecondOrder()
# geoOCCmesh.Refine()
# geoOCCmesh.Refine()