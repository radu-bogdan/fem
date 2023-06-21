import numba as nb
import numpy as np
from numba import float64,float32

##########################################################################################
@nb.njit() 
def f_nonlinear(x,y):
    k1 = 49.4; k2 = 1.46; k3 = 520.6
    return 1/(2*k2)*(np.exp(k2*(x**2+y**2))-1) + 1/2*k3*(x**2+y**2)

@nb.njit() 
def nu(x,y):
    k1 = 49.4; k2 = 1.46; k3 = 520.6
    return k1*np.exp(k2*(x**2+y**2))+k3

@nb.njit() 
def nux(x,y):
    k1 = 49.4; k2 = 1.46; k3 = 520.6
    return 2*x*k1*k2*np.exp(k2*(x**2+y**2))

@nb.njit() 
def nuy(x,y):
    k1 = 49.4; k2 = 1.46; k3 = 520.6
    return 2*y*k1*k2*np.exp(k2*(x**2+y**2))

@nb.njit() 
def fx_nonlinear(x,y):
    return nu(x,y)*x

@nb.njit() 
def fy_nonlinear(x,y):
    return nu(x,y)*y

@nb.njit() 
def fxx_nonlinear(x,y):
    return nu(x,y) + x*nux(x,y)

@nb.njit() 
def fxy_nonlinear(x,y):
    return x*nuy(x,y)

@nb.njit() 
def fyx_nonlinear(x,y):
    return y*nux(x,y)

@nb.njit() 
def fyy_nonlinear(x,y):
    return nu(x,y) + y*nuy(x,y)
##########################################################################################


# IDEE: vectorize???

##########################################################################################
@nb.njit((float64[:],float64[:]),fastmath=False, parallel=True, cache=True)
def gx_gy_nonlinear(x,y):
    
    le = x.size
    
    Bn = np.zeros((2,le),float64)
    
    for k in nb.prange(le):
        # Bnk = np.array([1,1],float64)
        Hnk = np.array([x[k],y[k]],float64)
        Bnk = np.linalg.norm(Bn[:,0])*Hnk/np.linalg.norm(Hnk)
        
        for it in range(10000):
            fxx = fxx_nonlinear(Bnk[0],Bnk[1]); fxy = fxy_nonlinear(Bnk[0],Bnk[1])
            fyx = fyx_nonlinear(Bnk[0],Bnk[1]); fyy = fyy_nonlinear(Bnk[0],Bnk[1])
            fx = fx_nonlinear(Bnk[0],Bnk[1]);   fy = fy_nonlinear(Bnk[0],Bnk[1])
            
            inv_Fxx = -1/(fxx*fyy-fxy*fyx)*np.array([[fyy,-fxy],[-fyx,fxx]])
            Fx = np.array([fx,fy])
            
            alpha = 1
            
            # if np.linalg.norm(Hnk-Fx,np.inf)<1e-12*(np.linalg.norm(Fx)+1):
            #     if alpha<1e-4:
            #         print(alpha)
            #     # print(k,it)
            #     break
            
            w = inv_Fxx@(Hnk-Fx)
            
            for r in range(1000):
                fxu = fx_nonlinear(Bnk[0]-alpha*w[0],Bnk[1]-alpha*w[1])
                fyu = fy_nonlinear(Bnk[0]-alpha*w[0],Bnk[1]-alpha*w[1])
                
                if (fxu-Hnk[0])**2+(fyu-Hnk[1])**2<=(fx-Hnk[0])**2+(fy-Hnk[1])**2: break                    
                else: alpha = alpha*(1/2)
                    
            
            Bnk1 = Bnk - alpha*w
            
            if np.linalg.norm(Bnk1-Bnk,np.inf)<1e-5:
                break
            
            Bnk = Bnk1.copy()
            
            if it==(10000-1):
                print("did not converge!")
        Bn[:,k] = Bnk1
    return Bn

##########################################################################################
def g_nonlinear_all(x,y):
    gx,gy = gx_gy_nonlinear(x,y)
    g = gx*x+gy*y-f_nonlinear(gx,gy)
    
    fxx = fxx_nonlinear(gx,gy)
    fxy = fxy_nonlinear(gx,gy)
    fyx = fyx_nonlinear(gx,gy)
    fyy = fyy_nonlinear(gx,gy)
    det = 1/(fxx*fyy-fyx*fxy)
    
    gxx =  det*fyy; gxy = -det*fxy
    gyx = -det*fyx; gyy =  det*fyy
    
    return g,gx,gy,gxx,gxy,gyx,gyy
##########################################################################################

import time
# a = np.random.rand(1_000_000)
# b = np.random.rand(1_000_000)

a = np.random.randint(100_000, size = 10_000_000).astype(float)
b = np.random.randint(100_000, size = 10_000_000).astype(float)

tm = time.monotonic(); g,gx,gy,gxx,gxy,gyx,gyy = g_nonlinear_all(a,b); print(time.monotonic()-tm)

for i in range(10000):
    a = np.random.randint(100_000, size = 10_000_000).astype(float)
    b = np.random.randint(100_000, size = 10_000_000).astype(float)
    tm = time.monotonic(); gx,gy = gx_gy_nonlinear(a,b); print(time.monotonic()-tm)

##########################################################################################

    err = np.linalg.norm((fx_nonlinear(gx,gy)-a)**2+\
                         (fy_nonlinear(gx,gy)-b)**2)
    print("inverse g' of f'. err: ",err)
    if err>1:
        break
##########################################################################################



