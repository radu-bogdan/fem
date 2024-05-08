import ngsolve as ngs
import netgen.occ as occ
# import netgen.gui
from netgen.webgui import Draw as DrawGeo
from ngsolve.webgui import Draw
import time

# import netgen.gui
# from netgen.webgui import Draw as DrawGeo

tm = time.monotonic()

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

##########################################################################
# Adding the steel parts
##########################################################################

coil_full = (box1-box2)+corner1_int-corner1_ext+corner2_int-corner2_ext+corner3_int-corner3_ext+corner4_int-corner4_ext

# mid_steel = occ.Box(occ.Pnt(-0.0016,-0.025,-0.0642),occ.Pnt(0.0016,0.025,0.0642))
mid_steel = occ.Box(occ.Pnt(-0.0016,-0.025,-(0.050+0.010)),occ.Pnt(0.0016,0.025,0.050+0.010))


r_steel1 = occ.Box(occ.Pnt(0.0016 +0.0005,0.015,0.050+0.010-0.0032),occ.Pnt(0.0016 +0.1205,0.065, 0.050+0.010))
r_steel2 = occ.Box(occ.Pnt(0.0016 +0.0005,0.015,-(0.050+0.010-0.0032)),occ.Pnt(0.0016 +0.1205,0.065,-(0.050+0.010)))
r_steel3 = occ.Box(occ.Pnt(0.0016 +0.1205-0.0032,0.015,0.050+0.010-0.0032),occ.Pnt(0.0016 +0.1205,0.065,-(0.050+0.010)))
r_steel = r_steel1 + r_steel2 + r_steel3

l_steel1 = occ.Box(occ.Pnt(-0.0016 -0.0005,-0.015, 0.050+0.010-0.0032),occ.Pnt(-0.0016-0.1205,-0.065,0.050+0.010))
l_steel2 = occ.Box(occ.Pnt(-0.0016 -0.0005,-0.015,-(0.050+0.010-0.0032)),occ.Pnt(-0.0016-0.1205,-0.065,-(0.050+0.010)))
l_steel3 = occ.Box(occ.Pnt(-0.0016 -0.1205 +0.0032,-0.015,0.050+0.010-0.0032),occ.Pnt(-0.0016-0.1205,-0.065,-(0.050+0.010)))
l_steel = l_steel1 + l_steel2 + l_steel3

##########################################################################
# Glueing ...
##########################################################################

half_box_1 = occ.Box(occ.Pnt(-0.100,-0.100,-0.050), occ.Pnt(0,0.100,0.050))
half_box_2 = occ.Box(occ.Pnt(0.100,-0.100,-0.050), occ.Pnt(0,0.100,0.050))

coil_half_box_1 = coil_full*half_box_1
coil_half_box_2 = coil_full*half_box_2

coil = occ.Glue([coil_half_box_1,coil_half_box_2])
ambient =  occ.Box(occ.Pnt(-0.200,-0.200,-0.100), occ.Pnt(0.200,0.200,0.100))

full = occ.Glue([coil, mid_steel, r_steel, l_steel, ambient])


##########################################################################
# Identifications
##########################################################################


for face in coil.faces: face.name = 'coil_face'
for face in r_steel.faces: face.name = 'r_steel_face'
for face in l_steel.faces: face.name = 'l_steel_face'
for face in mid_steel.faces: face.name = 'mid_steel_face'
for face in ambient.faces: face.name = 'ambient_face'

# steel_h = 0.002

coil_up_indices = (1,14)
coil_down_indices = (3,27)
coil_outer_indices = (0,2,4,5,13,19,18,17,16,15)
coil_inner_indices = (7,8,9,10,11,20,21,22,23,24,25)

for i in list(coil_outer_indices): coil.faces[i].name = 'coil_outer'
for i in list(coil_inner_indices): coil.faces[i].name = 'coil_inner'
for i in list(coil_up_indices): coil.faces[i].name = 'coil_up'
for i in list(coil_down_indices): coil.faces[i].name = 'coil_down'

coil.faces[6].name = 'coil_cut_1'
coil.faces[12].name = 'coil_cut_2'

print(coil.faces[6].mass)

coil.mat("coil")
r_steel.mat("r_steel")
l_steel.mat("l_steel")
mid_steel.mat("mid_steel")
ambient.mat("ambient")

h = 2**-5

steel_hf = h/8
for face in r_steel.faces: face.maxh = steel_hf
for face in l_steel.faces: face.maxh = steel_hf
for face in mid_steel.faces: face.maxh = steel_hf

steel_h = h/16

for edge in r_steel.edges: edge.maxh = steel_h
for edge in l_steel.edges: edge.maxh = steel_h
for edge in mid_steel.edges: edge.maxh = steel_h

# vh = h/100
# for vertex in r_steel.vertices : vertex.maxh = vh
# for vertex in l_steel.vertices : vertex.maxh = vh
# for vertex in mid_steel.vertices : vertex.maxh = vh


# ambient.maxh = 0.05
# coil.maxh = 0.05

##########################################################################
# "Fancy" coloring cuz why not I got a bit bored :)
##########################################################################

coil.faces.col = (1,0.5,0)
for i in list(coil_outer_indices): coil.faces[i].col = (0.5,0.7,1)
for i in list(coil_inner_indices): coil.faces[i].col = (0.5,0.7,0.4)
for i in list(coil_down_indices): coil.faces[i].col = (0.5,0.3,0.4)
for i in list(coil_up_indices): coil.faces[i].col = (0.5,0.1,0.7)

l_steel.faces.col=(1,0.5,1)
r_steel.faces.col=(1,0.5,1)
mid_steel.faces.col=(1,0.5,1)
ambient.faces.col=(1,1,1)

##########################################################################
# Generating mesh...
##########################################################################

geoOCC = occ.OCCGeometry(full)
# ng.Draw(geoOCC)
print('Geometry ... %.2fs'%(time.monotonic()-tm))

tm = time.monotonic()
geoOCCmesh = geoOCC.GenerateMesh(maxh = h)
print('Mesh ... %.2fs' %(time.monotonic()-tm))

##########################################################################

print(geoOCCmesh.ne)
# DrawGeo(full, clipping={"z":-1, "dist":0.08})
# Draw(geoOCCmesh, clipping={"z":-1, "dist":0.08})

geoOCCmesh.Save('whatever.vol')
mesh = ngs.Mesh('whatever.vol')
