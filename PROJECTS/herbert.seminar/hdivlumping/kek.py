from time import sleep
from ngsolve.fem import CompilePythonModule
from pathlib import Path
from ngsolve import *
from netgen.geom2d import unit_square
from netgen import geom2d
from ngsolve.webgui import Draw


def UnitTrig():
    geo = geom2d.SplineGeometry()
    p1 = geo.AppendPoint (0,0)
    p2 = geo.AppendPoint (1,0)
    p3 = geo.AppendPoint (0,1)
    geo.Append (["line", p1, p2])
    geo.Append (["line", p2, p3])
    geo.Append (["line", p3, p1])
    return Mesh(geo.GenerateMesh(maxh=1))

def TrigA():
    geo = geom2d.SplineGeometry()
    p1 = geo.AppendPoint (0,0)
    p2 = geo.AppendPoint (1,0)
    p3 = geo.AppendPoint (1,1)
    geo.Append (["line", p1, p2])
    geo.Append (["line", p2, p3])
    geo.Append (["line", p3, p1])
    return Mesh(geo.GenerateMesh(maxh=1))
def TrigB():
    geo = geom2d.SplineGeometry()
    p1 = geo.AppendPoint (1,0)
    p2 = geo.AppendPoint (1,1)
    p3 = geo.AppendPoint (0,1)
    geo.Append (["line", p1, p2])
    geo.Append (["line", p2, p3])
    geo.Append (["line", p3, p1])
    return Mesh(geo.GenerateMesh(maxh=1))

def UnitSquare():
    geo = geom2d.SplineGeometry()
    p1 = geo.AppendPoint (-1,-1)
    p2 = geo.AppendPoint (1,-1)
    p3 = geo.AppendPoint (1,1)
    p4 = geo.AppendPoint (-1,1)
    geo.Append (["line", p1, p2])
    geo.Append (["line", p2, p3])
    geo.Append (["line", p3, p4])
    geo.Append (["line", p4, p1])
    return Mesh(geo.GenerateMesh (maxh=0.1))

def TwoTrigs(_maxh=1, antidiagonal=False):
    geo = geom2d.SplineGeometry()
    p1 = geo.AppendPoint (0,0)
    p2 = geo.AppendPoint (1,0)
    p3 = geo.AppendPoint (0,1)
    p4 = geo.AppendPoint (1,1)
    geo.Append (["line", p1, p2])
    if antidiagonal:
        geo.Append (["line", p2, p3], leftdomain=1,rightdomain=1)
    geo.Append (["line", p3, p1])
    geo.Append (["line", p2, p4])
    geo.Append (["line", p4, p3])
    return Mesh(geo.GenerateMesh(maxh=_maxh))

mm = CompilePythonModule(Path('mymodule.cpp'), init_function_name='mymodule')