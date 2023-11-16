import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numba as nb
from numba import float64,float32

# bosch = True

# mu0 = 4*np.pi*10**-7

# if bosch == False:
# H_KL2 = np.array([0, 4.514672686, 16.55379985, 22.57336343, 28.59292701, 34.61249059, 37.62227239, 46.65161776, 49.66139955, 52.67118134, 55.68096313, 58.69074492, 61.70052671, 64.7103085, 67.72009029, 70.72987208, 73.73965388, 79.75921746, 82.76899925, 88.78856283, 91.79834462, 97.8179082, 103.8374718, 112.8668172, 121.8961625, 130.9255079, 142.9646351, 151.9939804, 167.0428894, 182.0917983, 209.1798345, 236.2678706, 260.3461249, 293.4537246, 326.5613243, 371.7080512, 419.8645598, 474.0406321, 540.2558315, 615.5003762, 702.7840482, 811.1361926, 901.4296464, 1015.801354, 1202.407825, 1346.877351, 1455.229496, 1668.924003, 1840.481565, 2135.440181, 2340.105342, 2592.927013, 2749.435666, 3116.629044, 3279.157261, 3513.920241, 3784.800602, 4097.817908, 4362.678706, 4663.656885, 4814.145974, 5000.752445, 5241.534989, 5416.102333, 5602.708804, 5771.256584, 6075.244545, 6409.330324, 6752.445448, 7101.580135, 7513.920241, 7829.947329, 8158.013544, 8365.688488, 1.28203768e+04, 1.65447489e+04, 2.07163957e+04, 2.55500961e+04, 3.15206135e+04, 4.03204637e+04, 7.73038295e+04, 1.29272791e+05, 1.81241752e+05, 2.33210713e+05, 2.85179674e+05, 3.37148635e+05, 3.89117596e+05, 4.41086557e+05, 4.93055518e+05, 5.45024479e+05, 5.96993440e+05, 6.48962401e+05, 7.00931362e+05, 7.52900323e+05, 8.04869284e+05, 8.56838245e+05, 9.08807206e+05, 9.60776167e+05, 1.01274513e+06, 1.06471409e+06, 1.11668305e+06, 1.16865201e+06, 1.22062097e+06, 1.27258993e+06, 1.32455889e+06, 1.37652785e+06, 1.42849682e+06, 1.48046578e+06, 1.53243474e+06, 1.58440370e+06, 1.63637266e+06, 1.68834162e+06, 1.74031058e+06, 1.79227954e+06, 1.84424850e+06, 1.89621746e+06, 1.94818643e+06, 2.00015539e+06, 2.05212435e+06, 2.10409331e+06, 2.15606227e+06, 2.20803123e+06, 2.26000019e+06] )
# B_KL2 = np.array([0, 0.007959134, 0.027873457, 0.043806714, 0.059739972, 0.083641356, 0.111528304, 0.151364444, 0.235028284, 0.270883359, 0.304746402, 0.346577572, 0.392392807, 0.434223977, 0.482031243, 0.529838509, 0.577645775, 0.63341967, 0.685210999, 0.754929117, 0.812696543, 0.86050231, 0.908308077, 0.964080473, 1.03180506, 1.077617296, 1.11944397, 1.159280111, 1.19512919, 1.234962333, 1.270805416, 1.3066485, 1.328548859, 1.352436754, 1.372340585, 1.39223842, 1.410142725, 1.424059968, 1.439963247, 1.45586203, 1.471754817, 1.48564508, 1.49356824, 1.505463473, 1.523298829, 1.535179073, 1.541101208, 1.556931042, 1.566805764, 1.582595127, 1.59046133, 1.606271678, 1.610177799, 1.625931189, 1.629834312, 1.639677558, 1.649502816, 1.663291154, 1.669135347, 1.678945617, 1.680862704, 1.6887379, 1.694594084, 1.700483244, 1.704374376, 1.710266534, 1.718083273, 1.725885023, 1.733682276, 1.743468564, 1.749239311, 1.757050054, 1.764854802, 1.768735442, 1.86530612e+00, 1.93061224e+00, 1.99591837e+00, 2.06122449e+00, 2.12653061e+00, 2.19183673e+00, 2.25714286e+00, 2.32244898e+00, 2.38775510e+00, 2.45306122e+00, 2.51836735e+00, 2.58367347e+00, 2.64897959e+00, 2.71428571e+00, 2.77959184e+00, 2.84489796e+00, 2.91020408e+00, 2.97551020e+00, 3.04081633e+00, 3.10612245e+00, 3.17142857e+00, 3.23673469e+00, 3.30204082e+00, 3.36734694e+00, 3.43265306e+00, 3.49795918e+00, 3.56326531e+00, 3.62857143e+00, 3.69387755e+00, 3.75918367e+00, 3.82448980e+00, 3.88979592e+00, 3.95510204e+00, 4.02040816e+00, 4.08571429e+00, 4.15102041e+00, 4.21632653e+00, 4.28163265e+00, 4.34693878e+00, 4.41224490e+00, 4.47755102e+00, 4.54285714e+00, 4.60816327e+00, 4.67346939e+00, 4.73877551e+00, 4.80408163e+00, 4.86938776e+00, 4.93469388e+00, 5.00000000e+00])
    
#     cut = 20

#     H_KL  = np.r_[H_KL[0],H_KL[cut:]]
#     B_KL  = np.r_[B_KL[0],B_KL[cut:]]
    
#     H_KL = H_KL.astype(float)
# else:
    
# H_KL = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 500000])
# B_KL = np.array([0,  0.076361010000000,   0.151802010000000,   0.224483020000000,   0.294404020000000,   0.360645030000000,   0.422286030000000,   0.479327040000000,   0.532688040000000,   0.581449050000000,   0.626530050000000,   0.913580110000000,    1.047910160000000,   1.123360210000000,   1.172130270000000,   1.205260320000000,   1.231030370000000,  1.250360420000000,   1.266010480000000,   1.279820530000000,   1.356281060000000,   1.400541590000000,   1.436522120000000,   1.467902650000000,   1.497443190000000,   1.525143720000000,   1.551004250000000,   1.575944780000000,   1.599045310000000,   1.761970620000000,   1.836575930000000,   1.870701240000000,   1.891026550000000,   1.905831860000000,   1.919717170000000,   1.932682480000000,   1.945647790000000,   1.958613100000000,   2.084586190000000,   2.461585480000000])
    
#     H_KL = H_KL[:-2]
#     B_KL = B_KL[:-2]
    
#     H_KL = H_KL.astype(float)
    
#     # H_KL_mess = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 500000])
#     # B_KL_mess = np.array([0,  0.076361010000000,   0.151802010000000,   0.224483020000000,   0.294404020000000,   0.360645030000000,   0.422286030000000,   0.479327040000000,   0.532688040000000,   0.581449050000000,   0.626530050000000,   0.913580110000000,    1.047910160000000,   1.123360210000000,   1.172130270000000,   1.205260320000000,   1.231030370000000,  1.250360420000000,   1.266010480000000,   1.279820530000000,   1.356281060000000,   1.400541590000000,   1.436522120000000,   1.467902650000000,   1.497443190000000,   1.525143720000000,   1.551004250000000,   1.575944780000000,   1.599045310000000,   1.761970620000000,   1.836575930000000,   1.870701240000000,   1.891026550000000,   1.905831860000000,   1.919717170000000,   1.932682480000000,   1.945647790000000,   1.958613100000000,   2.084586190000000,   2.461585480000000])


nu0 = 10**7/(4*np.pi)
k1 = 49.4; k2 = 1.46; k3 = 520.6


@nb.njit() 
def f_nonlinear(x,y):
    nu_nl = lambda x : k1*np.exp(k2*x)+k3
    pos = 1/k2*np.log(nu0/(k1*k2))
    m = pos-1/nu0*nu_nl(pos)
    return (k1/(2*k2)*(np.exp(k2*(x**2+y**2))-1) + 1/2*k3*(x**2+y**2))*(x<pos)+nu0*(1/2*(x**2+y**2))*(x>pos)



@nb.njit()
def nu(x):
    nu_nl = lambda x : k1*np.exp(k2*x)+k3
    pos = 1/k2*np.log(nu0/(k1*k2))
    m = pos-1/nu0*nu_nl(pos)
    return nu_nl(x)*(x<pos)+nu0*(x-m)*(x>pos)



@nb.njit()
def dx_nu(x):
    dx_nu_nl = lambda x : k1*k2*np.exp(k2*x)
    pos = 1/k2*np.log(nu0/(k1*k2))
    return dx_nu_nl(x)*(x<pos)+nu0*(x>pos)



# B_KL = np.exp(np.linspace(0,3,100))-1
# # B_KL = np.linspace(0,3,1000)
# H_KL = nu(B_KL)
# H_KL = H_KL.astype(float)


# BH_curve = interpolate.CubicSpline(B_KL, H_KL, bc_type='natural') # nu
# energy_density = BH_curve.antiderivative()
# dx_BH_curve = BH_curve.derivative() #nus

# def f_nonlinear(x,y):
#     return 1/2*energy_density(x**2+y**2)

# def fx_nonlinear(x,y):
#     return BH_curve(x**2+y**2)*x

# def fy_nonlinear(x,y):
#     return BH_curve(x**2+y**2)*y

# def fxx_nonlinear(x,y):
#     return BH_curve(x**2+y**2)+2*x*dx_BH_curve(x**2+y**2)*x

# def fxy_nonlinear(x,y):
#     return 2*x*y*dx_BH_curve(x**2+y**2)

# def fyx_nonlinear(x,y):
#     return 2*x*y*dx_BH_curve(x**2+y**2)

# def fyy_nonlinear(x,y):
#     return BH_curve(x**2+y**2)+2*y*dx_BH_curve(x**2+y**2)*y

@nb.njit() 
def fx_nonlinear(x,y):
    return nu(x**2+y**2)*x

@nb.njit() 
def fy_nonlinear(x,y):
    return nu(x**2+y**2)*y

@nb.njit() 
def fxx_nonlinear(x,y):
    return nu(x**2+y**2) + 2*x*dx_nu(x**2+y**2)*x

@nb.njit() 
def fxy_nonlinear(x,y):
    return 2*x*y*dx_nu(x**2+y**2)

@nb.njit() 
def fyx_nonlinear(x,y):
    return 2*x*y*dx_nu(x**2+y**2)

@nb.njit() 
def fyy_nonlinear(x,y):
    return nu(x**2+y**2) + 2*y*dx_nu(x**2+y**2)*y

f_linear = lambda x,y : 1/2*nu0*(x**2+y**2)
fx_linear = lambda x,y : nu0*x
fy_linear = lambda x,y : nu0*y
fxx_linear = lambda x,y : nu0 + 0*x
fxy_linear = lambda x,y : x*0
fyx_linear = lambda x,y : y*0
fyy_linear = lambda x,y : nu0 + 0*y

g_linear = lambda x,y : 1/(2*nu0)*(x**2+y**2)
gx_linear = lambda x,y : 1/nu0*x
gy_linear = lambda x,y : 1/nu0*y
gxx_linear = lambda x,y : 1/nu0 + 0*x
gxy_linear = lambda x,y : x*0
gyx_linear = lambda x,y : y*0
gyy_linear = lambda x,y : 1/nu0 + 0*y




##########################################################################################

# def g_nonlinear_all(x,y):
    
#     g_nl = g_nonlinear(x,y)
#     gx_nl = gx_nonlinear(x,y)
#     gy_nl = gy_nonlinear(x,y)
    
#     gxx_nl = gxx_nonlinear(x,y)
#     gxy_nl = gxy_nonlinear(x,y)
#     gyx_nl = gyx_nonlinear(x,y)
#     gyy_nl = gyy_nonlinear(x,y)
    
#     g_l = g_linear(x,y)
#     gx_l = gx_linear(x,y)
#     gy_l = gy_linear(x,y)
    
#     gxx_l = gxx_linear(x,y)
#     gxy_l = gxy_linear(x,y)
#     gyx_l = gyx_linear(x,y)
#     gyy_l = gyy_linear(x,y)
    
#     return g_l, gx_l, gy_l, gxx_l, gxy_l, gyx_l, gyy_l,\
#            g_nl,gx_nl,gy_nl,gxx_nl,gxy_nl,gyx_nl,gyy_nl
           
##########################################################################################

# plt.figure()
# xx = np.exp(np.linspace(0,4,100_000_00))
# plt.plot(xx,BH_curve(xx),'-')
# plt.plot(B_KL,H_KL,'*')

# plt.figure()
# xx = np.linspace(5.70000000e+02,7_000_000,1_000_000)
# plt.plot(xx,dx_HB_curve(xx),'-')
# plt.plot(H_KL,B_KL,'*')


@nb.njit((float64[:],float64[:]), fastmath=True, cache=True)
# from numpy import float64

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
    gx_nl,gy_nl = gx_gy_nonlinear(x,y)
    g_nl = gx_nl*x+gy_nl*y-f_nonlinear(gx_nl,gy_nl)
    
    fxx = fxx_nonlinear(gx_nl,gy_nl)
    fxy = fxy_nonlinear(gx_nl,gy_nl)
    fyx = fyx_nonlinear(gx_nl,gy_nl)
    fyy = fyy_nonlinear(gx_nl,gy_nl)
    inv_det = 1/(fxx*fyy-fyx*fxy)
    
    gxx_nl =  inv_det*fyy; gxy_nl = -inv_det*fxy
    gyx_nl = -inv_det*fyx; gyy_nl =  inv_det*fxx
    
    g_l = g_linear(x,y)
    gx_l = gx_linear(x,y)
    gy_l = gy_linear(x,y)
    
    gxx_l = gxx_linear(x,y)
    gxy_l = gxy_linear(x,y)
    gyx_l = gyx_linear(x,y)
    gyy_l = gyy_linear(x,y)
    
    return g_l, gx_l, gy_l, gxx_l, gxy_l, gyx_l, gyy_l,\
           g_nl,gx_nl,gy_nl,gxx_nl,gxy_nl,gyx_nl,gyy_nl
           
##########################################################################################