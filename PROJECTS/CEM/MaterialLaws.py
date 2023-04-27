# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:15:52 2023

@author: Michael
"""
import numpy as np
from scipy import optimize


def Brauer(k1,k2,k3):
    f= lambda x,y: k1/2/k2*(np.exp(k2*x**2+k2*y**2)-1)+1/2*k3*(x**2+y**2)
    
    fx= lambda x,y: (k1*np.exp(k2*(x**2+y**2))+k3)*x
    fy= lambda x,y: (k1*np.exp(k2*(x**2+y**2))+k3)*y
    
    fxx=lambda x,y: k1*np.exp(k2*(x**2+y**2))+2*x**2*k1*k2*np.exp(k2*(x**2+y**2))+k3
    fyy=lambda x,y: k1*np.exp(k2*(x**2+y**2))+2*y**2*k1*k2*np.exp(k2*(x**2+y**2))+k3
    fxy=fyx=lambda x,y: 2*x*y*k1*k2*np.exp(k2*(x**2+y**2))
    df= lambda x,y:np.array([fx(x,y),fy(x,y)])
    ddf= lambda x,y:np.array([[fxx(x,y),fxy(x,y)],[fyx(x,y),fyy(x,y)]])
    return f,df,ddf

def LinMaterialG(nuMaterial):    
    f=lambda x,y: nuMaterial/2*(x**2+y**2)
    
    fx=lambda x,y: nuMaterial*x
    fy=lambda x,y: nuMaterial*y
    fxx=lambda x,y: nuMaterial+x*0
    fyy=lambda x,y: nuMaterial+x*0
    fxy=fyx=lambda x,y: 0+x*0
    df= lambda x,y,:np.array ([fx(x,y),fy(x,y)])
    ddf= lambda x,y:np.array([[fxx(x,y),fxy(x,y)],[fyx(x,y),fyy(x,y)]])
    return f,df,ddf

def LinMaterialE(nuMaterial,nu0):    
    fac=nuMaterial/(nu0-nuMaterial)
    print(fac,10)
    f=lambda x,y: fac/2*(x**2+y**2)
    
    fx=lambda x,y: fac*x
    fy=lambda x,y: fac*y
    fxx=lambda x,y: fac+x*0
    fyy=lambda x,y: fac+x*0
    fxy=fyx=lambda x,y: 0+x*0
    df= lambda x,y,:np.array ([fx(x,y),fy(x,y)])
    ddf= lambda x,y:np.array([[fxx(x,y),fxy(x,y)],[fyx(x,y),fyy(x,y)]])
    return f,df,ddf

def HerbertsMaterialG(a,b,nu0,n):
    normxyz=lambda x,y,z,n: (((x**2)+(y**2))**n+(z**2)**n)**(1/2/n)
    
    f=lambda x,y: nu0/2*(x**2+y**2)-a*normxyz(x,y,b,n)+0*x#wrong
     
    fx=lambda x,y: nu0*x- a*x/normxyz(x,y,b,n)+0*x
    fy=lambda x,y: nu0*y- a*y/normxyz(x,y,b,n)+0*x
    
    fxx=lambda x,y: nu0-a/normxyz(x,y,b,n)+a*(x**2)*(x**2+y**2)**(n-1)/normxyz(x, y, b,n)**(2*n+1)
    fyy=lambda x,y: nu0-a/normxyz(x,y,b,n)+a*(y**2)*(x**2+y**2)**(n-1)/normxyz(x, y, b,n)**(2*n+1)
    fyx=lambda x,y: a*(x*y)*(x**2+y**2)**(n-1)/normxyz(x, y, b,n)**(2*n+1)+0*x
    fxy=lambda x,y: a*(x*y)*(x**2+y**2)**(n-1)/normxyz(x, y, b,n)**(2*n+1)+0*x
    df= lambda x,y:np.array ([fx(x,y),fy(x,y)]) 
    ddf= lambda x,y:np.array([[fxx(x,y),fxy(x,y)],[fyx(x,y),fyy(x,y)]])
    return f,df,ddf

def HerbertsMaterialE(a,b,nu0,n):
    sqrtTerm=lambda x,y: (((a/nu0)**2)**n-(x**2+y**2)**n)**(1/2/n)
    
    f=lambda x,y:-1/2*(x**2+y**2)-b*np.sqrt((a/nu0)**2-(x**2+y**2))
    
    fx=lambda x,y: -x+ b*x/sqrtTerm(x,y)
    fy=lambda x,y: -y+ b*y/sqrtTerm(x,y)
    
    print((a/nu0)**2,100000)
    fxx=lambda x,y: -1+b/sqrtTerm(x,y)+b*x**2*(x**2+y**2)**(n-1)/sqrtTerm(x,y)**(2*n+1)
    fyy=lambda x,y: -1+b/sqrtTerm(x,y)+b*y**2*(x**2+y**2)**(n-1)/sqrtTerm(x,y)**(2*n+1)    
    fyx=fxy=lambda x,y: b*(x*y)*(x**2+y**2)**(n-1)/sqrtTerm(x,y)**(2*n+1) 
    df= lambda x,y:np.array ([fx(x,y),fy(x,y)])
    ddf= lambda x,y:np.array([[fxx(x,y),fxy(x,y)],[fyx(x,y),fyy(x,y)]])
    return f,df,ddf