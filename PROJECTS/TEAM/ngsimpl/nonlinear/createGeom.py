import netgen.occ as occ

def makeGeo():
    r_coil = 0.025
    y_coil = 0.05
    r_outer = 0.1
    height = 0.012

    print("dsadas313")

    ###############################################################
    c1 = occ.Cylinder((0,0,0), occ.Z, r = r_outer, h = height)
    c2 = occ.Cylinder((0,+y_coil,0), occ.Z, r = r_coil, h = height)
    c3 = occ.Cylinder((0,-y_coil,0), occ.Z, r = r_coil, h = height)

    full = occ.Glue([c1,c2,c3])

    full.solids[0].name = "stator"
    full.solids[1].name = "coil_minus"
    full.solids[2].name = "coil_plus"

    full.faces[0].name = 'outer_face'
    full.faces[1].name = 'stator_face'
    full.faces[2].name = 'stator_face'
    full.faces[3].name = 'f3'
    full.faces[4].name = 'f4'
    full.faces[5].name = 'f5'
    full.faces[6].name = 'coil_minus_face'
    full.faces[7].name = 'coil_minus_face'
    full.faces[8].name = 'f8'
    full.faces[9].name = 'coil_plus_face'
    full.faces[10].name = 'coil_plus_face'

    full.solids[0].name = "stator"
    full.solids[1].name = "coil_minus"
    full.solids[2].name = "coil_plus"

    geoOCC = occ.OCCGeometry(full)
    ###############################################################

    return geoOCC