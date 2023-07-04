# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:15:52 2023

@author: Michael
"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
# import mygrad as mg


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

def biro(b1,b2,db1,ddb1,db2,ddb2,n):    
    # F=lambda x,y,w:(x/b1(w))**n+(y/b2(w))**n-1
    # dF= lambda x,y,w:-n*((x/b1(w))**(n-1)*(x*db1(w))/(b1(w))**2+(y/b2(w))**(n-1)*(y*db2(w))/(b2(w))**2)
    n_=lambda n,x: (x**2)**(n/2)
    
    F1=lambda x,y,w:((x/b1(w))**2)**(n/2)     +     ((y/b2(w))**2)**(n/2)      -1
    dF1= lambda x,y,w:-n*(       ((x/b1(w))**2)**(n/2)*db1(w)/b1(w)              +          ((y/b2(w))**2)**(n/2)*db2(w)/b2(w)         )
    
    # Fx_=lambda x,y,w:n*(x/b1(w))**n/x
    # Fy_=lambda x,y,w:n*(y/b2(w))**n/y
    # Fz_=lambda x,y,w:n*(-db1(w)*(x/b1(w))**n/b1(w)-db2(w)*(y/b2(w))**n/b2(w))
    
    # Fx_=lambda x,y,w:n*((x/b1(w))**2)**(n/2)/x
    # Fy_=lambda x,y,w:n*((y/b2(w))**2)**(n/2)/y
    
    Fx_=lambda x,y,w:n*(x**2)**(n/2-0.5)/(b1(w)**2)**(n/2)*np.sign(x)
    Fy_=lambda x,y,w:n*(y**2)**(n/2-0.5)/(b2(w)**2)**(n/2)*np.sign(y)
    
    Fz_=lambda x,y,w:n*(-db1(w)*((x/b1(w))**2)**(n/2)/b1(w)-db2(w)*((y/b2(w))**2)**(n/2)/b2(w))


    Fxx_ = lambda x,y,z: (((n - 1) * n * (x / b1(z)) ** (n - 2)) / b1(z) ** 2)
    Fyy_ = lambda x,y,z: (((n - 1) * n * (y / b2(z)) ** (n - 2)) / b2(z) ** 2)
    Fxz_ = lambda x,y,z: (-((n - 1) * n * x * db1(z) * (x / b1(z)) ** (n - 2)) / b1(z) ** 3) - ((n * db1(z) * (x / b1(z)) ** (n - 1)) / b1(z) ** 2)
    Fyz_ = lambda x,y,z: (-((n - 1) * n * y * db2(z) * (y / b2(z)) ** (n - 2)) / b2(z) ** 3) - ((n * db2(z) * (y / b2(z)) ** (n - 1)) / b2(z) ** 2)
    Fzz_ = lambda x,y,z:     -(n * x * ddb1(z) * (x / b1(z)) **           (n - 1)) /       b1(z) ** 2 +   (((n - 1) * n * x * x * db1(z) ** 2 *  (x / b1(z)) **        (n - 2)) /       b1(z) ** 4) +  ((2 * n * x * db1(z) ** 2 * (x / b1(z)) **         (n - 1)) /     b1(z) ** 3) -    (n * y * ddb2(z) *   (y / b2(z)) **        (n - 1)) /      b2(z) ** 2 +    (((n - 1) * n * y * y * db2(z) ** 2 * (y / b2(z)) **        (n - 2)) /         b2(z) ** 4) + ((2 * n * y * db2(z) ** 2 * (y / b2(z)) **         (n - 1)) / b2(z) ** 3)


    # Fxx_ = lambda x, y, z: (((n - 1) * n * ((x / b1(z)) ** 2) ** ((n - 2) / 2)) / (b1(z) ** 2))
    # Fyy_ = lambda x, y, z: (((n - 1) * n * ((y / b2(z)) ** 2) ** ((n - 2) / 2)) / (b2(z) ** 2))
    # Fxz_ = lambda x, y, z: (-((n - 1) * n * x * db1(z) * ((x / b1(z)) ** 2) ** ((n - 2) / 2)) / (b1(z) ** 3)) - ((n * db1(z) * ((x / b1(z)) ** 2) ** ((n - 1) / 2)) / (b1(z) ** 2))
    # Fyz_ = lambda x, y, z: (-((n - 1) * n * y * db2(z) * ((y / b2(z)) ** 2) ** ((n - 2) / 2)) / (b2(z) ** 3)) - ((n * db2(z) * ((y / b2(z)) ** 2) ** ((n - 1) / 2)) / (b2(z) ** 2))
    # Fzz_ = lambda x, y, z: (-((n * x * ddb1(z) * ((x / b1(z)) ** 2) ** ((n - 1) / 2)) / (b1(z) ** 2))) + (((n - 1) * n * x * x * db1(z) ** 2 * ((x / b1(z)) ** 2) ** ((n - 2) / 2)) / (b1(z) ** 4)) + ((2 * n * x * db1(z) ** 2 * ((x / b1(z)) ** 2) ** ((n - 1) / 2)) / (b1(z) ** 3)) - ((n * y * ddb2(z) * ((y / b2(z)) ** 2) ** ((n - 1) / 2)) / (b2(z) ** 2)) + (((n - 1) * n * y * y * db2(z) ** 2 * ((y / b2(z)) ** 2) ** ((n - 2) / 2)) / (b2(z) ** 4)) + ((2 * n * y * db2(z) ** 2 * ((y / b2(z)) ** 2) ** ((n - 1) / 2)) / (b2(z) ** 3))
   


    def f(x,y):
        w=np.linalg.norm(np.array([x,y]));
        F_=lambda w:F1(x,y,w)
        dF_=lambda w:dF1(x,y,w)
        res=np.abs(F_(w))
        for k in range(50):
            res=np.abs(F_(w))
            if res<1.e-10:
                    return abs(w)
            sx=-F_(w)/dF_(w)
            alpha =1
            for m in range(1000):
                if np.abs(F_(w+alpha*sx))<(1-0.000001*alpha)*res:
                    break
                else:
                    alpha=alpha/2
            w=w+alpha*sx    
            w=np.abs(w)
        return 0
    def df(x,y):
        w= f(x,y)
        # fx=lambda x,y: (x/b1(w))**(n-1)/b1(w)/((x/b1(w))**n*(db1(w)/b1(w))+(y/b2(w))**n*(db2(w)/b2(w)))
        # fy=lambda x,y: (y/b2(w))**(n-1)/b2(w)/((x/b1(w))**n*(db1(w)/b1(w))+(y/b2(w))**n*(db2(w)/b2(w)))
        
        fx=lambda x,y: -Fx_(x,y,w)/Fz_(x,y,w)
        fy=lambda x,y: -Fy_(x,y,w)/Fz_(x,y,w)
        
        return np.array([fx(x,y),fy(x,y)])
    def ddf(x,y):
        sigx=np.sign(x)
        sigy=np.sign(y)
        x=abs(x)
        y=abs(y)
        w=f(x,y)
        g=df(x,y)
        gx=g[0]
        gy=g[1]     
        Fx=Fx_(x,y,w)
        Fy=Fy_(x,y,w)
        Fz=Fz_(x,y,w)

        Fxx=Fxx_(x,y,w)
        Fyy=Fyy_(x,y,w)
        Fyz=Fyz_(x,y,w)
        Fxz=Fxz_(x,y,w)
        Fzz=Fzz_(x,y,w)
        fxx=lambda x,y:(-(Fxx+Fxz*gx)*Fz+(Fxz+Fzz*gx)*Fx)/(Fz**2)
        fyy=lambda x,y:(-(Fyy+Fyz*gy)*Fz+(Fyz+Fzz*gy)*Fy)/(Fz**2)
        fxy=lambda x,y:(-(Fxz*gy)*Fz+(Fyz+Fzz*gy)*Fx)/(Fz**2)
        fyx=lambda x,y:(-(Fyz*gx)*Fz+(Fxz+Fzz*gx)*Fy)/(Fz**2)

        return np.array([[fxx(x,y),sigx*sigy*fxy(x,y)],[sigx*sigy*fyx(x,y),fyy(x,y)]])
    
        
    return f,df,ddf

def transform(r,phi):
    return np.array([r*np.cos(phi),r*np.sin(phi)])


def biroTest(fac,n):
    # b1= lambda x:x**2
    # db1=lambda x:2*x
    # ddb1=lambda y:2
    # b2= lambda x:x**2
    # db2=lambda x:2*x
    # ddb2=lambda x:2
    
    b1= lambda x:np.arctan(x)
    db1=lambda x:1/(x**2+1)
    ddb1=lambda x:-(2*x)/(1 + x**2)**2
    b2= lambda x:np.arctan(fac*x)
    db2=lambda x:fac/(fac*fac*x**2+1)
    ddb2=lambda x:-(2*fac*fac*fac*x)/(1 + fac*fac*x**2)**2
    return biro(b1,b2,db1,ddb1,db2,ddb2,n)

def birotestcurve(fac,n):
    b,db,ddb=biroTest(fac,n)
    sample_phi = np.array(range(1000))
    pl=[];
    fail=0
    print("llllarge")
    for k in range(8000):
        sample=transform(1,k/1000)
        if b(sample[0],sample[1])!=0:
            pl.append(db(sample[0],sample[1])) 
            if(np.isnan(ddb(sample[0],sample[1])).any()):
                print(sample[0],sample[1])
                fail=fail+1
    print("large")
    for k in range(8000):
        sample=transform(1,k/1000)
        if b(sample[0],sample[1])!=0:
            pl.append(db(sample[0],sample[1]))
            if(np.isnan(ddb(sample[0],sample[1])).any()):
                print(sample[0],sample[1])
                print(b(sample[0],sample[1]))
                fail=fail+1
    print("mittel")
    for k in range(8000):
        sample=transform(.7,k/1000)
        if b(sample[0],sample[1])!=0:
            pl.append(db(sample[0],sample[1]))
            if(np.isnan(ddb(sample[0],sample[1])).any()):
                print(sample[0],sample[1])
                fail=fail+1
    for k in range(8000):
        sample=transform(.6,k/1000)
        if b(sample[0],sample[1])!=0:
            pl.append(db(sample[0],sample[1]))
            if(np.isnan(ddb(sample[0],sample[1])).any()):
                print(sample[0],sample[1])
                fail=fail+1
    for k in range(8000):
        sample=transform(.5,k/1000)
        if b(sample[0],sample[1])!=0:
            pl.append(db(sample[0],sample[1]))
            if(np.isnan(ddb(sample[0],sample[1])).any()):
                print(sample[0],sample[1])
                fail=fail+1
    print("hello")
    for k in range(8000):
        sample=transform(.1,k/1000)
        if b(sample[0],sample[1])!=0:
            pl.append(db(sample[0],sample[1]))
            if(np.isnan(ddb(sample[0],sample[1])).any()):
                print(sample[0],sample[1])        
                fail=fail+1
    print("middle")
    for k in range(8000):
        sample=transform(.4,k/1000)
        if b(sample[0],sample[1])!=0:
            pl.append(db(sample[0],sample[1]))
            if(np.isnan(ddb(sample[0],sample[1]).any())):
                print(sample[0],sample[1])
                fail=fail+1
    print("small")
    for k in range(8000):
        sample=transform(.001,k/1000)
        if b(sample[0],sample[1])!=0:
            pl.append(db(sample[0],sample[1]))
            if(np.isnan(ddb(sample[0],sample[1])).any()):
                print(sample[0],sample[1])
                fail=fail+1
                
    
    pl=np.array(pl)
    print(fail)
    plt.scatter(pl[:,0],pl[:,1],s=.05)

def birotestcurveContur(fac,n):
    b,db,ddb=biroTest(fac,n)#

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    Xm, Ym = np.meshgrid(x, y)
    Z=np.zeros(100*100)
    Z2=np.zeros(100*100)
    for k in range(100):
        for m in range(100): 
            Z2[(k)*100+m]=b(k/100-0.5+0.00001,m/100-0.5+.00001)
    Z2=Z2.reshape((100,100))
    plt.contour(Xm, Ym, Z2, colors='black');
    plt.pause(0.05)
    
    