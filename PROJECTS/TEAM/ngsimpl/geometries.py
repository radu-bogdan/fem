# def Make2dGeometry(nr):
#     if nr==2:
#         return Make2dGeometry2()
#     else:
#         print("geometry", nr, "not defined")
#         return 0


# simple geometry of transformer core
# similar to friedrich 2019, but with corners 
def Make2dGeometry1():
    print("making transformer 2d geometry")
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()

    mm=0.001
    
    h1=2
    # point coordinates: x, y, maxh ...
    pnts = [(0,0), (13,0), (8,0), (28,0), (33,0), (40,0), (40,40), (0,40), (0,28), (0,18), 
           (13,10), (18,10), (28,10), (33,10), (18,18), (28,28)]
    # pnts = [(0,0), (15,0), (18,0), (28,0), (31,0), (40,0), (40,40), (0,40), (0,28), (0,18), 
    #        (15,10), (18,10), (28,10), (31,10), (18,18), (28,28)]
    
    # pnums = [geo.AppendPoint(*p) for p in pnts]
    for x,y in pnts: 
        geo.AppendPoint(x*mm,y*mm, maxh=h1*mm)
    
    # start-point, end-point, boundary-condition, domain left, domain right, maxh
    lines = [(0,1,1,1,0), (1,2,1,3,0), (2,3,1,2,0), (3,4,1,4,0), (4,5,1,1,0), 
             (5,6,2,1,0), (6,7,2,1,0), (7,8,1,1,0), (8,9,1,2,0), (9,0,1,1,0), 
             (1,10,3,1,3), (10,11,3,1,3), (11,2,3,2,3), (3,12,3,2,4), (12,13,3,1,4), 
             (13,4,3,1,4), (11,14,3,1,2), (14,9,3,1,2), (12,15,3,2,1), (15,8,3,2,1)] 
        
    for p1,p2,bc,left,right in lines:
        # geo.Append( ["line", pnums[p1], pnums[p2]], bc=bc, leftdomain=left, rightdomain=right, maxh=h)
        geo.Append( ["line", p1, p2], bc=bc, leftdomain=left, rightdomain=right, maxh=h1)

    # don't know if this works??
    geo.SetMaterial(1, "air")
    geo.SetMaterial(2, "core")
    geo.SetMaterial(3, "coil1")
    geo.SetMaterial(4, "coil2")

    return geo


# Benchmark Geometry 2 from Friedrich 2019
def Make2dGeometry2():
    print("making benchmark geometry 2 of Friedrich 2019 paper")
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()

    mm=0.001
    
    h1=2
    # point coordinates: x, y, maxh ...
    pnts = [(0,0), (13,0), (18,0), (28,0), (33,0), (40,0), (40,40), (0,40), (0,28), (0,18), 
            (13,10), (18,10), (28,10), (33,10), (18,16), (18,18), (16,18), (28,23), (28,28), (23,28)]
    
    # pnums = [geo.AppendPoint(*p) for p in pnts]
    for x,y in pnts: 
        geo.AppendPoint(x*mm,y*mm, maxh=h1*mm)
    
    # start-point, end-point, boundary-condition, domain left, domain right, maxh
    lines = [(0,1,1,1,0), (1,2,1,3,0), (2,3,1,2,0), (3,4,1,4,0), (4,5,1,1,0), 
             (5,6,2,1,0), (6,7,2,1,0), (7,8,1,1,0), (8,9,1,2,0), (9,0,1,1,0), 
             (1,10,3,1,3), (10,11,3,1,3), (11,2,3,2,3), (3,12,3,2,4), (12,13,3,1,4), (13,4,3,1,4),
             (11,14,3,1,2), (16,9,3,1,2), (12,17,3,2,1), (19,8,3,2,1)] 
        
    for p1,p2,bc,left,right in lines:
        # geo.Append( ["line", pnums[p1], pnums[p2]], bc=bc, leftdomain=left, rightdomain=right, maxh=h)
        geo.Append( ["line", p1, p2], bc=bc, leftdomain=left, rightdomain=right, maxh=h1)

    # start, mid, end, bc, left, right 
    splines = [(14,15,16,3,1,2), (17,18,19,3,2,1)]

    for p1,p2,p3,bc,left,right in splines:
        # geo.Append( ["spline", p1, p2, bc=bc, leftdomain=left, rightdomain=right, maxh=h)
        geo.Append   (["spline3",p1,p2,p3], bc=bc, leftdomain=left, rightdomain=right, maxh=h1)

    # don't know if this works??
    geo.SetMaterial(1, "air")
    geo.SetMaterial(2, "core")
    geo.SetMaterial(3, "coil1")
    geo.SetMaterial(4, "coil2")

    return geo



##############################################################################
def Make3dGeometry1():
    print("making team13 geometry")
    import netgen.occ as occ
    mm=0.001

    box1 = occ.Box(occ.Pnt(-100*mm,-100*mm,-50*mm), occ.Pnt(100*mm,100*mm,50*mm))
    box2 = occ.Box(occ.Pnt(-75*mm,-75*mm,-50*mm), occ.Pnt(75*mm,75*mm,50*mm))

    ##########################################################################
    # Rounding corners ...
    ##########################################################################

    corner1_ext = occ.Box(occ.Pnt(75*mm,75*mm,-50*mm), occ.Pnt(100*mm,100*mm,50*mm))
    cyl1_ext = occ.Cylinder(occ.Pnt(75*mm,75*mm,-50*mm), occ.Z, r=25*mm, h=100*mm)
    corner1_int = occ.Box(occ.Pnt(50*mm,50*mm,-50*mm), occ.Pnt(75*mm,75*mm,50*mm))
    cyl1_int = occ.Cylinder(occ.Pnt(50*mm,50*mm,-50*mm), occ.Z, r=25*mm, h=100*mm)
    corner1_int = corner1_int-cyl1_int; corner1_ext = corner1_ext-cyl1_ext

    corner2_ext = occ.Box(occ.Pnt(-100*mm,-100*mm,-50*mm), occ.Pnt(-75*mm,-75*mm,50*mm))
    cyl2_ext = occ.Cylinder(occ.Pnt(-75*mm,-75*mm,-50*mm), occ.Z, r=25*mm, h=100*mm)
    corner2_int = occ.Box(occ.Pnt(-75*mm,-75*mm,-50*mm), occ.Pnt(-50*mm,-50*mm,50*mm))
    cyl2_int = occ.Cylinder(occ.Pnt(-50*mm,-50*mm,-50*mm), occ.Z, r=25*mm, h=100*mm)
    corner2_int = corner2_int-cyl2_int; corner2_ext = corner2_ext-cyl2_ext

    corner3_ext = occ.Box(occ.Pnt(75*mm,-75*mm,-50*mm), occ.Pnt(100*mm,-100*mm,50*mm))
    cyl3_ext = occ.Cylinder(occ.Pnt(75*mm,-75*mm,-50*mm), occ.Z, r=25*mm, h=100*mm)
    corner3_int = occ.Box(occ.Pnt(50*mm,-50*mm,-50*mm), occ.Pnt(75*mm,-75*mm,50*mm))
    cyl3_int = occ.Cylinder(occ.Pnt(50*mm,-50*mm,-50*mm), occ.Z, r=25*mm, h=100*mm)
    corner3_int = corner3_int-cyl3_int; corner3_ext = corner3_ext-cyl3_ext

    corner4_ext = occ.Box(occ.Pnt(-75*mm,75*mm,-50*mm), occ.Pnt(-100*mm,100*mm,50*mm))
    cyl4_ext = occ.Cylinder(occ.Pnt(-75*mm,75*mm,-50*mm), occ.Z, r=25*mm, h=100*mm)
    corner4_int = occ.Box(occ.Pnt(-50*mm,50*mm,-50*mm), occ.Pnt(-75*mm,75*mm,50*mm))
    cyl4_int = occ.Cylinder(occ.Pnt(-50*mm,50*mm,-50*mm), occ.Z, r=25*mm, h=100*mm)
    corner4_int = corner4_int-cyl4_int; corner4_ext = corner4_ext-cyl4_ext

    ##########################################################################
    # Adding the steel parts
    ##########################################################################

    coil_full = (box1-box2)+corner1_int-corner1_ext+corner2_int-corner2_ext+corner3_int-corner3_ext+corner4_int-corner4_ext

    mid_steel = occ.Box(occ.Pnt(-1.6*mm,-25*mm,-64.2*mm),occ.Pnt(1.6*mm,25*mm,64.2*mm))

    r_steel1 = occ.Box(occ.Pnt(1.6*mm+0.5*mm,15*mm,50*mm+10*mm-3.2*mm),occ.Pnt(1.6*mm+120.5*mm,65*mm,50*mm+10*mm))
    r_steel2 = occ.Box(occ.Pnt(1.6*mm+0.5*mm,15*mm,-(50*mm+10*mm-3.2*mm)),occ.Pnt(1.6*mm+120.5*mm,65*mm,-(50*mm+10*mm)))
    r_steel3 = occ.Box(occ.Pnt(1.6*mm+120.5*mm-3.2*mm,15*mm,50*mm+10*mm-3.2*mm),occ.Pnt(1.6*mm+120.5*mm,65*mm,-(50*mm+10*mm)))
    r_steel = r_steel1 + r_steel2 + r_steel3

    l_steel1 = occ.Box(occ.Pnt(-1.6*mm-0.5*mm,-15*mm,50*mm+10*mm-3.2*mm),occ.Pnt(-1.6*mm-120.5*mm,-65*mm,50*mm+10*mm))
    l_steel2 = occ.Box(occ.Pnt(-1.6*mm-0.5*mm,-15*mm,-(50*mm+10*mm-3.2*mm)),occ.Pnt(-1.6*mm-120.5*mm,-65*mm,-(50*mm+10*mm)))
    l_steel3 = occ.Box(occ.Pnt(-1.6*mm-120.5*mm+3.2*mm,-15*mm,50*mm+10*mm-3.2*mm),occ.Pnt(-1.6*mm-120.5*mm,-65*mm,-(50*mm+10*mm)))
    l_steel = l_steel1 + l_steel2 + l_steel3

    ##########################################################################
    # Glueing ...
    ##########################################################################

    half_box_1 = occ.Box(occ.Pnt(-100*mm,-100*mm,-50*mm), occ.Pnt(0*mm,100*mm,50*mm))
    half_box_2 = occ.Box(occ.Pnt(100*mm,-100*mm,-50*mm), occ.Pnt(0*mm,100*mm,50*mm))

    coil_half_box_1 = coil_full*half_box_1
    coil_half_box_2 = coil_full*half_box_2

    coil = occ.Glue([coil_half_box_1,coil_half_box_2])
    ambient =  occ.Box(occ.Pnt(-200*mm,-200*mm,-100*mm), occ.Pnt(200*mm,200*mm,100*mm))

    full = occ.Glue([coil, mid_steel, r_steel, l_steel, ambient])
    # full = occ.Glue([coil, mid_steel, r_steel, l_steel])

    ##########################################################################
    # "Fancy" coloring cuz why not I got a bit bored :)
    ##########################################################################

    coil.faces.col=(1,0.5,0)
    l_steel.faces.col=(1,0.5,1)
    r_steel.faces.col=(1,0.5,1)
    mid_steel.faces.col=(1,0.5,1)
    ambient.faces.col=(1,1,1)

    ##########################################################################
    # Identifications
    ##########################################################################


    for face in coil.faces: face.name = 'coil_face'
    for face in r_steel.faces: face.name = 'r_steel_face'
    for face in l_steel.faces: face.name = 'l_steel_face'
    for face in mid_steel.faces: face.name = 'mid_steel_face'
    for face in ambient.faces: face.name = 'ambient_face'

    coil.faces[6].name = 'coil_cut_1'
    coil.faces[12].name = 'coil_cut_2'

    coil.mat("coil")
    r_steel.mat("steel")
    l_steel.mat("steel")
    mid_steel.mat("steel")
    ambient.mat("air")

    geoOCC = occ.OCCGeometry(full)
    return geoOCC
