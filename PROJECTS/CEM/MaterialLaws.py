# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:15:52 2023

@author: Michael
"""
import numpy as np

def nonlinearMaterial1():
    f = lambda x,y : x**2 + y**2 + 0.1*x**4 + 0.1*y**4
    
    fx = lambda x,y : 2*x+y+0.4*x**3
    fy = lambda x,y : 2*y+x+0.4*y**3
    
    fxx = lambda x,y : 2+1.2*x**4
    fxy = lambda x,y : 1+0*x
    fyx = lambda x,y : 1+0*x
    fyy = lambda x,y : 2+1.2*y**4

    return f,fx,fy,fxx,fxy,fyx,fyy

def HerbertsMaterialG(a,b):
    normxyz=lambda x,y,z: np.linalg.norm(np.array([x,y,z])) 
    
    f=lambda x,y,nu0:nu0/2*(x**2+y**2)+a*normxyz(x,y,b)
    
    fx=lambda x,y,nu0: nu0*x+ a*x/normxyz(x,y,b)
    fy=lambda x,y,nu0: nu0*y+ a*y/normxyz(x,y,b)
    
    fxx=lambda x,y,nu0: nu0+a/normxyz(x,y,b)-a*x**2/normxyz(x, y, b)/(x**2+y**2+b**2)
    fyy=lambda x,y,nu0: nu0+a/normxyz(x,y,b)-a*y**2/normxyz(x, y, b)/(x**2+y**2+b**2)
    fxy=fyx=lambda x,y,nu0: a*x*y/normxyz(x, y, b)/(x**2+y**2+b**2)
    df= lambda x,y,nu0:np.array ([fx(x,y,nu0),fy(x,y,nu0)])
    ddf= lambda x,y,nu0:np.array([[fxx(x,y,nu0),fxy(x,y,nu0)],[fyx(x,y,nu0),fyy(x,y,nu0)]])
    return f,df,ddf

def HerbertsMaterialE(a,b):
    Nenner=lambda x,y,a: np.sqrt(a**2-x**2-y**2)
    
    f=lambda x,y,nu0: nu0/2*(x**2+y**2)-b*Nenner(x*nu0,y*nu0,a)
    
    fx=lambda x,y,nu0: nu0*x+ nu0**2*b*x/Nenner(x*nu0,y*nu0,a)
    fy=lambda x,y,nu0: nu0*y+ nu0**2*b*y/Nenner(x*nu0,y*nu0,a)
    
    fxx=lambda x,y,nu0: nu0+b*nu0**2/Nenner(x*nu0,y*nu0,a)+b*nu0**4*x**2/Nenner(x*nu0,y*nu0,a)**3
    fyy=lambda x,y,nu0: nu0+b*nu0**2*Nenner(x*nu0,y*nu0,a)+b*nu0**4*y**2/Nenner(x*nu0,y*nu0,a)**3
    fxy=fyx=lambda x,y,nu0: b*nu0**4*x*y/Nenner(x*nu0,y*nu0,a)**3
    df= lambda x,y,nu0:npp.array ([fx(x,y,nu0),fy(x,y,nu0)])
    ddf= lambda x,y,nu0:np.array([[fxx(x,y,nu0),fxy(x,y,nu0)],[fyx(x,y,nu0),fyy(x,y,nu0)]])
    return f,df,ddf
