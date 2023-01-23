#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 12:06:03 2023

@author: catalinradu
"""

import gmsh

def unitSquare():
    # # Unit square
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(1, 0, 0)
    p3 = gmsh.model.geo.addPoint(1, 1, 0)
    p4 = gmsh.model.geo.addPoint(0, 1, 0)
    
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    c1 = gmsh.model.geo.addCurveLoop([1, 2, 3, 4])
    s1 = gmsh.model.geo.addPlaneSurface([c1])
    
    gmsh.model.addPhysicalGroup(1, [l1])
    gmsh.model.addPhysicalGroup(1, [l2])
    gmsh.model.addPhysicalGroup(1, [l3])
    gmsh.model.addPhysicalGroup(1, [l4])

def capacitorPlates(a,b,c,d,l):
    p1 = gmsh.model.geo.addPoint(-a/2, -b/2, 0)
    p2 = gmsh.model.geo.addPoint( a/2, -b/2, 0)
    p3 = gmsh.model.geo.addPoint( a/2,  b/2, 0)
    p4 = gmsh.model.geo.addPoint(-a/2,  b/2, 0)
    
    p5 = gmsh.model.geo.addPoint(-c-d/2, -l/2, 0)
    p6 = gmsh.model.geo.addPoint(  -d/2, -l/2, 0)
    p7 = gmsh.model.geo.addPoint(  -d/2,  l/2, 0)
    p8 = gmsh.model.geo.addPoint(-c-d/2,  l/2, 0)
    
    p9 = gmsh.model.geo.addPoint(   d/2, -l/2, 0)
    p10= gmsh.model.geo.addPoint( c+d/2, -l/2, 0)
    p11= gmsh.model.geo.addPoint( c+d/2,  l/2, 0)
    p12= gmsh.model.geo.addPoint(   d/2,  l/2, 0)
    
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p5)
    
    l9 = gmsh.model.geo.addLine(p9, p10)
    l10= gmsh.model.geo.addLine(p10, p11)
    l11= gmsh.model.geo.addLine(p11, p12)
    l12= gmsh.model.geo.addLine(p12, p9)
    
    l13= gmsh.model.geo.addLine(p6, p9)
    l14= gmsh.model.geo.addLine(p7, p12)
    
    c1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    c2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
    c3 = gmsh.model.geo.addCurveLoop([l9, l10, l11, l12])
    c4 = gmsh.model.geo.addCurveLoop([l13, -l12, -l14, -l6])
    
    gmsh.model.geo.addPlaneSurface([c1,c2,c3,c4])
    gmsh.model.geo.addPlaneSurface([c4])
    # gmsh.model.geo.addPlaneSurface([c1,c4])
    
    gmsh.model.geo.synchronize()
    
    gmsh.model.addPhysicalGroup(1, [l1])
    gmsh.model.addPhysicalGroup(1, [l2])
    gmsh.model.addPhysicalGroup(1, [l3])
    gmsh.model.addPhysicalGroup(1, [l4])
    
    gmsh.model.addPhysicalGroup(1, [l5])
    gmsh.model.addPhysicalGroup(1, [l6])
    gmsh.model.addPhysicalGroup(1, [l7])
    gmsh.model.addPhysicalGroup(1, [l8])
    
    gmsh.model.addPhysicalGroup(1, [l9])
    gmsh.model.addPhysicalGroup(1, [l10])
    gmsh.model.addPhysicalGroup(1, [l11])
    gmsh.model.addPhysicalGroup(1, [l12])
    
    gmsh.model.geo.synchronize()
    
    

def geometryP2():
    p1 = gmsh.model.geo.addPoint( 0,  0, 0)
    p2 = gmsh.model.geo.addPoint(12,  0, 0)
    p3 = gmsh.model.geo.addPoint(12, 11, 0)
    p4 = gmsh.model.geo.addPoint( 0, 11, 0)
    
    p5 = gmsh.model.geo.addPoint( 2,  2, 0)
    p6 = gmsh.model.geo.addPoint(10,  2, 0)
    p7 = gmsh.model.geo.addPoint(10,  3, 0)
    p8 = gmsh.model.geo.addPoint( 8,  3, 0)
    p9 = gmsh.model.geo.addPoint( 4,  3, 0)    
    p10= gmsh.model.geo.addPoint( 2,  3, 0)
    
    p11= gmsh.model.geo.addPoint( 2, 4, 0)
    p12= gmsh.model.geo.addPoint( 4, 4, 0)
    p13= gmsh.model.geo.addPoint( 8, 4, 0)
    p14= gmsh.model.geo.addPoint(10, 4, 0)
    
    p15= gmsh.model.geo.addPoint( 2, 9, 0)
    p16= gmsh.model.geo.addPoint(10, 9, 0)
    
    p17= gmsh.model.geo.addPoint( 4, 7, 0)
    p18= gmsh.model.geo.addPoint( 8, 7, 0)
    
    p19= gmsh.model.geo.addPoint(4.5, 9.5, 0)
    p20= gmsh.model.geo.addPoint(7.5, 9.5, 0)
    p21= gmsh.model.geo.addPoint(7.5, 9, 0)
    p22= gmsh.model.geo.addPoint(4.5, 9, 0)
    
    p23= gmsh.model.geo.addPoint(4.5, 7, 0)
    p24= gmsh.model.geo.addPoint(7.5, 7, 0)
    p25= gmsh.model.geo.addPoint(7.5, 6.5, 0)
    p26= gmsh.model.geo.addPoint(4.5, 6.5, 0)
    
    # gmsh.model.geo.synchronize()
    
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p9)
    l9 = gmsh.model.geo.addLine(p9, p10)
    l10= gmsh.model.geo.addLine(p10, p5)
    
    l11= gmsh.model.geo.addLine(p10, p11)
    l12= gmsh.model.geo.addLine(p11, p12)
    l13= gmsh.model.geo.addLine(p12, p9)
    
    l14= gmsh.model.geo.addLine(p8, p13)
    l15= gmsh.model.geo.addLine(p13, p14)
    l16= gmsh.model.geo.addLine(p14, p7)
        
    l17= gmsh.model.geo.addLine(p11, p15)
    l18= gmsh.model.geo.addLine(p22, p21)
    l19= gmsh.model.geo.addLine(p16, p14)
    l20= gmsh.model.geo.addLine(p12, p17)
    l21= gmsh.model.geo.addLine(p23, p24)
    l22= gmsh.model.geo.addLine(p18, p13)
    
    l23= gmsh.model.geo.addLine(p19, p20)
    l24= gmsh.model.geo.addLine(p20, p21)
    # l25= gmsh.model.geo.addLine(p21, p22)
    l26= gmsh.model.geo.addLine(p22, p19)
    
    # l27= gmsh.model.geo.addLine(p23, p24)
    l28= gmsh.model.geo.addLine(p24, p25)
    l29= gmsh.model.geo.addLine(p25, p26)
    l30= gmsh.model.geo.addLine(p26, p23)
    
    l31= gmsh.model.geo.addLine(p17, p23)
    l32= gmsh.model.geo.addLine(p24, p18)
    
    l33= gmsh.model.geo.addLine(p15, p22)
    l34= gmsh.model.geo.addLine(p21, p16)
    
    gmsh.model.geo.synchronize()
    
    c1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    c2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8, l9, l10])
    c3 = gmsh.model.geo.addCurveLoop([l9, l11, l12, l13])
    c4 = gmsh.model.geo.addCurveLoop([l7, l14, l15, l16])
    c5 = gmsh.model.geo.addCurveLoop([-l12, l17, l33, l18, l34, l19, -l15, -l22, -l32, -l21, -l31, -l20])
    c6 = gmsh.model.geo.addCurveLoop([l8, -l13, l20, l31, -l30, -l29, -l28, l32, l22,-l14])
    
    c7 = gmsh.model.geo.addCurveLoop([l23, l24, -l18, l26])
    c8 = gmsh.model.geo.addCurveLoop([-l21, -l28, -l29, -l30])
    
    # gmsh.model.geo.synchronize()
    
    gmsh.model.geo.addPlaneSurface([c1,c2,c3,c4,c5,c6,c7,c8])
    
    gmsh.model.geo.addPlaneSurface([c5])
    gmsh.model.geo.addPlaneSurface([c2])
    gmsh.model.geo.addPlaneSurface([c3])
    gmsh.model.geo.addPlaneSurface([c4])
    gmsh.model.geo.addPlaneSurface([c6])
    
    
    gmsh.model.geo.addPlaneSurface([c7])
    gmsh.model.geo.addPlaneSurface([c8])
    
    gmsh.model.geo.synchronize()
    
    # gmsh.model.geo.synchronize()
    
    gmsh.model.addPhysicalGroup(dim = 1, tags = [l1])
    gmsh.model.addPhysicalGroup(dim = 1, tags = [l2])
    gmsh.model.addPhysicalGroup(dim = 1, tags = [l3])
    gmsh.model.addPhysicalGroup(dim = 1, tags = [l4])
    