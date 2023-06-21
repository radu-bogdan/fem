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


# # IDEE: vectorize???
# @nb.njit((float64[:],float64[:]),fastmath=True, cache=True)
# def gx_gy_nonlinear_vek(x,y):
#     le = x.size
#     Hn0 = x; Hn1 = y
#     Bn0 = np.zeros((1,le))+1; Bn1 = np.zeros((1,le))+1
    
#     for it in range(10000):
#         fxx = fxx_nonlinear(Bn0,Bn1); fxy = fxy_nonlinear(Bn0,Bn1)
#         fyx = fyx_nonlinear(Bn0,Bn1); fyy = fyy_nonlinear(Bn0,Bn1)
#         fx  = fx_nonlinear (Bn0,Bn1); fy  = fy_nonlinear (Bn0,Bn1)
        
#         inv_detF = 1/(fxx*fyy-fxy*fyx)
#         inv_F_xx =-inv_detF*fyy; inv_F_xy = inv_detF*fxy
#         inv_F_yx = inv_detF*fyx; inv_F_yy =-inv_detF*fxx
        
#         # w = inv_Fxx@(Hnk-Fx)
        
#         w0 = inv_F_xx*(Hn0-fx)+inv_F_xy*(Hn1-fy)
#         w1 = inv_F_yx*(Hn0-fx)+inv_F_yy*(Hn1-fy)
        
#         alpha = np.zeros((1,le))+1
        
#         for r in range(1000):
#             fxu = fx_nonlinear(Bn0-alpha*w0,Bn1-alpha*w1)
#             fyu = fy_nonlinear(Bn0-alpha*w0,Bn1-alpha*w1)
            
#             if ((((fxu-Hn0)**2+(fyu-Hn1)**2)-((fx-Hn0)**2+(fy-Hn1)**2))<0).all(): break
#             else: alpha = alpha*(1/2)
            
#         Bn0u = Bn0 - alpha*w0
#         Bn1u = Bn1 - alpha*w1
        
#         if (np.sqrt((Bn1u-Bn1)**2+(Bn0u-Bn0)**2)<1e-5).all():
#             # print(it)
#             break
        
#         Bn0 = Bn0u.copy()
#         Bn1 = Bn1u.copy()
        
#         if it==(10000-1):
#             print("did not converge!")
            
#     return Bn0,Bn1

##########################################################################################
@nb.njit((float64[:],float64[:]),fastmath=True, cache=True)
def gx_gy_nonlinear(x,y):
    
    le = x.size
    
    Bn = np.zeros((2,le),float64)
    
    for k in nb.prange(le):
        # Bnk = np.array([1,1],float64)
        Hnk = np.array([x[k],y[k]],float64)
        Bnk = np.linalg.norm(Bn[:,0])*Hnk/np.linalg.norm(Hnk)
        Hnk0 = x[k]; Hnk1 = y[k]
        Bnk0 = Bnk[0]; Bnk1 = Bnk[1];
        
        for it in range(10000):
            fxx = fxx_nonlinear(Bnk0,Bnk1); fxy = fxy_nonlinear(Bnk0,Bnk1)
            fyx = fyx_nonlinear(Bnk0,Bnk1); fyy = fyy_nonlinear(Bnk0,Bnk1)
            fx  = fx_nonlinear(Bnk0,Bnk1);  fy  = fy_nonlinear(Bnk0,Bnk1)            
            
            alpha = 1
            
            inv_detF = 1/(fxx*fyy-fxy*fyx)
            inv_F_xx =-inv_detF*fyy; inv_F_xy = inv_detF*fxy
            inv_F_yx = inv_detF*fyx; inv_F_yy =-inv_detF*fxx
            
            w0 = inv_F_xx*(Hnk0-fx)+inv_F_xy*(Hnk1-fy)
            w1 = inv_F_yx*(Hnk0-fx)+inv_F_yy*(Hnk1-fy)
        
            # w = inv_Fxx@(Hnk-Fx)
            
            for r in range(1000):
                fxu = fx_nonlinear(Bnk0-alpha*w0,Bnk1-alpha*w1)
                fyu = fy_nonlinear(Bnk0-alpha*w0,Bnk1-alpha*w1)
                
                if (fxu-Hnk0)**2+(fyu-Hnk1)**2<=(fx-Hnk0)**2+(fy-Hnk1)**2: break                    
                else: alpha = alpha*(1/2)
                    
            
            # Bnk1 = Bnk - alpha*np.array([w0,w1])
            Bnku0 = Bnk0 - alpha*w0
            Bnku1 = Bnk1 - alpha*w1
            # Bnk1 = Bnk - alpha*w
            
            # if np.linalg.norm(np.array([Bnk0,Bnk1])-np.array([Bnku0,Bnuk1])
            # if np.linalg.norm(Bnk1-Bnk,np.inf)<1e-3:
            if (Bnku0-Bnk0)**2+(Bnku1-Bnk1)**2<(1e-8)**2:
                # print(it)
                break
            
            # Bnk = Bnk1.copy()
            Bnk0 = Bnku0
            Bnk1 = Bnku1
            
            if it==(10000-1):
                print("did not converge!")
        Bn[0,k] = Bnk0
        Bn[1,k] = Bnk1
        # Bn[:,k] = Bnk1
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

# import time
# # a = np.random.rand(1_000_000)
# # b = np.random.rand(1_000_000)

# a = np.random.randint(100_000, size = 1_000).astype(float)
# b = np.random.randint(100_000, size = 1_000).astype(float)

# tm = time.monotonic(); g,gx,gy,gxx,gxy,gyx,gyy = g_nonlinear_all(a,b); print(time.monotonic()-tm)


# sample_size = 1_000_000
# for i in range(100):
#     a = np.random.randint(100_000, size = 10_000_000).astype(float)
#     b = np.random.randint(100_000, size = 10_000_000).astype(float)

##########################################################################################

    # tm = time.monotonic(); gx,gy = gx_gy_nonlinear(a,b); print('jit: ',time.monotonic()-tm)
    # err = np.linalg.norm((fx_nonlinear(gx,gy)-a)**2+\
    #                      (fy_nonlinear(gx,gy)-b)**2)/sample_size
    # print("inverse g' of f'. err: ",err)
    
##########################################################################################

##########################################################################################

    # tm = time.monotonic(); gx,gy = gx_gy_nonlinear_vek(a,b); print('vec: ',time.monotonic()-tm)
    # err = np.linalg.norm((fx_nonlinear(gx,gy)-a)**2+\
    #                      (fy_nonlinear(gx,gy)-b)**2)
    # print("inverse g' of f'. err: ",err)
    
##########################################################################################



