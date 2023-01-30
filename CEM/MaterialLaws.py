# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:15:52 2023

@author: Michael
"""
import numpy as np
from autograd import grad
from autograd import jacobian

def makeHandles(f):
    df=jacobian(f)
    ddf=jacobian(df)
    return df, ddf

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
    
    fxx=lambda x,y,nu0: nu0+a/normxyz(x,y,b)-x**2/normxyz(x, y, b)/(x**2+y**2+b**2)
    fyy=lambda x,y,nu0: nu0+a/normxyz(x,y,b)-y**2/normxyz(x, y, b)/(x**2+y**2+b**2)
    fxy=fyx=lambda x,y,nu0: a*x*y/normxyz(x, y, b)/(x**2+y**2+b**2)
    df= lambda x,y,nu0:np.array ([fx(x,y,nu0),fy(x,y,nu0)])
    ddf= lambda x,y,nu0:np.array([[fxx(x,y,nu0),fxy(x,y,nu0)],[fyx(x,y,nu0),fyy(x,y,nu0)]])
    return f,df,ddf

def HerbertsMaterialE(a,b):
    normxy=lambda x,y: np.linalg.norm(np.array([x,y])) 
    g,dg,ddg=HerbertsMaterialG(a,b)
    I_dg_inv_radial=lambda R,nu0: np.sqrt(R**2*b**2/((a/nu0)**2-R**2))
    de=lambda x,y,nu0:dg(I_dg_inv_radial(normxy(x,y),nu0)*x/normxy(x,y),I_dg_inv_radial(normxy(x,y),nu0)*y/normxy(x,y))
    
    def Hesse(x,y,nu0):
        dirx=x/normxy(x,y)
        diry=x/normxy(x,y)
        ru=I_dg_inv_radial(normxy(x,y),nu0)
        ddg_=ddg(ru*dirx,ru*diry,nu0)
        H=ddg_@ np.linalg.inv(np.eye(2,2)-1./nu0*ddg_)
        return H
    
    dde=lambda x,y,nu0:Hesse(x,y,nu0)
    
    return de, dde
    