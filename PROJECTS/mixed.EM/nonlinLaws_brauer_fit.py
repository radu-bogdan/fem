import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numba as nb
from numba import float64,float32

from scipy.optimize import curve_fit 

nu0 = 10**7/(4*np.pi)
k1 = 49.4; k2 = 1.46; k3 = 520.6
pos = np.sqrt(1/k2*np.log(1/k1*(nu0-k3)))

@nb.njit()
def nu(x):
    nu_nl = lambda x : (k1*np.exp(k2*x**2)+k3)
    pos = np.sqrt(1/k2*np.log(1/k1*(nu0-k3)))
    return nu_nl(x)*(x<pos)+nu0*(x>pos)+0*(np.exp(-(x-pos))**10)*(x>pos)

xx = np.linspace(0,4,300)
nu_spline = interpolate.CubicSpline(xx, nu(xx), bc_type = 'natural')
spl = interpolate.make_smoothing_spline(xx, nu_spline.derivative()(xx))
nu_derivative_spline = interpolate.BSpline(xx, spl(xx)*(spl(xx)>0),k=3)

xx = np.linspace(0,4,3000)
# plt.plot(xx,nu(xx),'.')
plt.plot(xx,nu_derivative_spline.antiderivative(0)(xx),'.')



# plt.plot(xx,nu_spline.derivative()(xx),'*')


# def func(x, a, b, c):
#     return a*x**3 + b*x**2 + c*x

# popt, pcov = curve_fit(func, xx, nu(xx))
# plt.plot(xx, func(xx, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


# @nb.njit()
# def dx_nu(x):
#     dx_nu_nl = lambda x : k1*k2*np.exp(k2*x)
#     pos = 1/k2*np.log(nu0/(k1*k2))
#     return dx_nu_nl(x)*(x<pos)+nu0*(x>pos)


# plt.figure()
xx = np.linspace(0,4,3000)
# plt.plot(xx,nu(xx),'.')
# plt.plot(1/2*(xx[1:]+xx[:-1]), nu_spline(xx[1:])-nu_spline(xx[:-1]),'.')
# plt.plot(xx, nu_spline.derivative()(xx),'.')
# plt.plot(xx, nu(xx),'.')
# plt.plot(B_KL,H_KL,'*')



# tck = interpolate.splrep(1/2*(xx[1:]+xx[:-1]), nu(xx[1:])-nu(xx[:-1]), k=3)
# ys_smooth = interpolate.splev(1/2*(xx[1:]+xx[:-1]), tck)
# ys_smooth = interpolate.BSpline(*tck)(nu(xx[1:])-nu(xx[:-1]))

# yy = interpolate.CubicSpline(1/2*(xx[1:]+xx[:-1]), nu(xx[1:])-nu(xx[:-1]), bc_type='natural')

xx = np.linspace(0,4,400)
# plt.plot(xx,nu_spline(xx),'*')
# plt.plot(1/2*(xx[1:]+xx[:-1]), nu_spline(xx[1:])-nu_spline(xx[:-1]),'+')
# plt.plot(xx, nu_spline.derivative()(xx),'+')


# plt.figure()
# xx = np.linspace(5.70000000e+02,7_000_000,1_000_000)
# plt.plot(xx,dx_HB_curve(xx),'-')
# plt.plot(H_KL,B_KL,'*')


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
#     gx_nl,gy_nl = gx_gy_nonlinear(x,y)
#     g_nl = gx_nl*x+gy_nl*y-f_nonlinear(gx_nl,gy_nl)
    
#     fxx = fxx_nonlinear(gx_nl,gy_nl)
#     fxy = fxy_nonlinear(gx_nl,gy_nl)
#     fyx = fyx_nonlinear(gx_nl,gy_nl)
#     fyy = fyy_nonlinear(gx_nl,gy_nl)
#     inv_det = 1/(fxx*fyy-fyx*fxy)
    
#     gxx_nl =  inv_det*fyy; gxy_nl = -inv_det*fxy
#     gyx_nl = -inv_det*fyx; gyy_nl =  inv_det*fxx
    
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