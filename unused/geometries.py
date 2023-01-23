#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 22:41:56 2022

@author: bogdan
"""

import gmsh

def generate_unit_square_01():

    gmsh.initialize()
    gmsh.finalize()
    gmsh.initialize()
    
    gmsh.model.geo.addPoint( 0, 0, 0, 0 ,1)
    gmsh.model.geo.addPoint( 1, 0, 0, 0 ,2)
    gmsh.model.geo.addPoint( 1, 1, 0, 0 ,3)
    gmsh.model.geo.addPoint( 0, 1, 0, 0 ,4)
    
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    
    gmsh.model.geo.synchronize()
    
    model = gmsh.model
    
    # return model