import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numba as nb
from numba import float64,float32
# from scipy.optimize import curve_fit 

nu0 = 10**7/(4*np.pi)
k1 = 49.4; k2 = 1.46; k3 = 520.6

# @nb.njit()
# def gen_nu(x):
#     nu_nl = lambda x : (k1*np.exp(k2*x**2)+k3)
#     pos = np.sqrt(1/k2*np.log(1/k1*(nu0-k3)))
#     return nu_nl(x)*(x<pos)+nu0*(x>pos)

@nb.njit()
def gen_nu(x):
    nu_nl = lambda x : (k1*np.exp(k2*x)+k3)
    pos = 1/k2*np.log(1/k1*(nu0-k3))
    return nu_nl(x)*(x<pos)+nu0*(x>pos)

xx = np.linspace(0,8,5000)
yy = np.exp(np.linspace(0,np.log(5*1e8),5000))-1

nu = interpolate.CubicSpline(xx, gen_nu(xx), bc_type = 'natural')
# spl = interpolate.make_smoothing_spline(xx, nu.derivative()(xx))
# dx_nu = interpolate.BSpline(xx, spl(xx)*(spl(xx)>0),k=1)
dx_nu = nu.derivative(1)

gen_gs = interpolate.CubicSpline(xx*nu(xx**2), xx, bc_type = 'natural')
mu = interpolate.CubicSpline(yy[1:]**2, gen_gs(yy[1:])/yy[1:], bc_type = 'natural')
dx_mu = mu.derivative(1)


# plt.close('all')

# xx = np.linspace(0,8,3000)
# plt.figure()
# plt.plot(xx,dx_nu.antiderivative(0)(xx),'.')
# plt.plot(xx,dx_nu.antiderivative(1)(xx),'.')
# plt.plot(xx,dx_nu.antiderivative(2)(xx),'.')
# plt.plot(xx,nu0+0*xx)


# yy = np.exp(np.linspace(0,np.log(1e8),300))-1
# plt.figure()
# plt.plot(yy,mu(yy),'*')
# plt.plot(yy,mu.derivative()(yy),'*')
# plt.plot(yy,1/nu0+0*yy)
# plt.plot(xx,dx_nu.antiderivative(0)(xx)*xx,'.')

# # check inverse!

# plt.figure()
# plt.plot(mu(yy**2)*nu(mu(yy**2)**2*yy**2))
# # plt.plot(nu(xx**2)*mu(nu(xx**2)**2*xx**2))

##########################################################################################

def f_nonlinear(x,y):
    return 1/2*nu.antiderivative(1)(x**2+y**2)

def fx_nonlinear(x,y):
    return nu(x**2+y**2)*x

def fy_nonlinear(x,y):
    return nu(x**2+y**2)*y

def fxx_nonlinear(x,y):
    return nu(x**2+y**2) + 2*x*dx_nu(x**2+y**2)*x

def fxy_nonlinear(x,y):
    return 2*x*y*dx_nu(x**2+y**2)

def fyx_nonlinear(x,y):
    return 2*x*y*dx_nu(x**2+y**2)

def fyy_nonlinear(x,y):
    return nu(x**2+y**2) + 2*y*dx_nu(x**2+y**2)*y

f_linear = lambda x,y : 1/2*nu0*(x**2+y**2)
fx_linear = lambda x,y : nu0*x
fy_linear = lambda x,y : nu0*y
fxx_linear = lambda x,y : nu0 + 0*x
fxy_linear = lambda x,y : x*0
fyx_linear = lambda x,y : y*0
fyy_linear = lambda x,y : nu0 + 0*y

##########################################################################################

def g_nonlinear(x,y):
    return 1/2*mu.antiderivative(1)(x**2+y**2)

def gx_nonlinear(x,y):
    return mu(x**2+y**2)*x

def gy_nonlinear(x,y):
    return mu(x**2+y**2)*y

def gxx_nonlinear(x,y):
    return mu(x**2+y**2) + 2*x*dx_mu(x**2+y**2)*x

def gxy_nonlinear(x,y):
    return 2*x*y*dx_mu(x**2+y**2)

def gyx_nonlinear(x,y):
    return 2*x*y*dx_mu(x**2+y**2)

def gyy_nonlinear(x,y):
    return mu(x**2+y**2) + 2*y*dx_mu(x**2+y**2)*y


g_linear = lambda x,y : 1/(2*nu0)*(x**2+y**2)
gx_linear = lambda x,y : 1/nu0*x
gy_linear = lambda x,y : 1/nu0*y
gxx_linear = lambda x,y : 1/nu0 + 0*x
gxy_linear = lambda x,y : x*0
gyx_linear = lambda x,y : y*0
gyy_linear = lambda x,y : 1/nu0 + 0*y


##########################################################################################

def g_nonlinear_all(x,y):
    
    g_nl = g_nonlinear(x,y)
    gx_nl = gx_nonlinear(x,y)
    gy_nl = gy_nonlinear(x,y)
    
    gxx_nl = gxx_nonlinear(x,y)
    gxy_nl = gxy_nonlinear(x,y)
    gyx_nl = gyx_nonlinear(x,y)
    gyy_nl = gyy_nonlinear(x,y)
    
    g_l = g_linear(x,y)
    gx_l = gx_linear(x,y)
    gy_l = gy_linear(x,y)
    
    gxx_l = gxx_linear(x,y)
    gxy_l = gxy_linear(x,y)
    gyx_l = gyx_linear(x,y)
    gyy_l = gyy_linear(x,y)
    
    return g_l, gx_l, gy_l, gxx_l, gxy_l, gyx_l, gyy_l,\
           g_nl,gx_nl,gy_nl,gxx_nl,gxy_nl,gyx_nl,gyy_nl
           
           