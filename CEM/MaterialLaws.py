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

def HerbertsMaterialG(a,b,nu0):
    normxyz=lambda x,y,z: np.linalg.norm(np.array([x,y,z])) 
    
    f=lambda x,y:nu0/2*(x**2+y**2)+a*normxyz(x,y,b)
    
    fx=lambda x,y: nu0*x+ a*x/normxyz(x,y,b)
    fy=lambda x,y: nu0*y+ a*y/normxyz(x,y,b)
    
    fxx=lambda x,y: nu0+a/normxyz(x,y,b)-x**2/normxyz(x, y, b)/(x**2+y**2+b**2)
    fyy=lambda x,y: nu0+a/normxyz(x,y,b)-y**2/normxyz(x, y, b)/(x**2+y**2+b**2)
    fxy=fyx=lambda x,y: a*x*y/normxyz(x, y, b)/(x**2+y**2+b**2)
    df= lambda x,y:np.array ([fx(x,y),fy(x,y)])
    ddf= lambda x,y:np.array([[fxx(x,y),fxy(x,y)],[fyx(x,y),fyy(x,y)]])
    return f,df,ddf

def HerbertsMaterialE(a,b,nu0):
    normxy=lambda x,y: np.linalg.norm(np.array([x,y])) 
    g,dg,ddg=HerbertsMaterialG(a,b,nu0)
    I_dg_inv_radial=lambda R: np.sqrt(R**2*b**2/((a/nu0)**2-R**2))
    dex=lambda x,y:dg(I_dg_inv_radial(normxy(x,y))*x/normxy(x,y))
    dey=lambda x,y:dg(I_dg_inv_radial(normxy(x,y))*y/normxy(x,y))
    
    de= lambda x,y:np.array ([dex(x,y),dey(x,y)])
    
    def Hesse(x,y):
        dirx=x/normxy(x,y)
        diry=x/normxy(x,y)
        ru=I_dg_inv_radial(normxy(x,y))
        ddg_=ddg(ru*dirx,ru*diry)
        H=ddg_@np.linalg.inv(np.eye(2,2)-1./nu0*ddg_)
        return H
    
    dde=lambda x,y:Hesse(x,y)
    
    return de, dde
    