import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numba as nb

nu0 = 10**7/(4*np.pi)
k1 = 49.4; k2 = 1.46; k3 = 520.6

@nb.njit()
def gen_nu(x):
    nu_nl = lambda x : (k1*np.exp(k2*x)+k3)
    pos = 1/k2*np.log(1/k1*(nu0-k3))
    return nu_nl(x)*(x<pos)+nu0*(x>pos)

xx = np.linspace(0,8,5000)
yy = np.exp(np.linspace(0,np.log(5*1e8),5000))-1

nu = interpolate.CubicSpline(xx, gen_nu(xx), bc_type = 'natural')
spl = interpolate.make_smoothing_spline(xx, nu.derivative()(xx))
dx_nu = interpolate.BSpline(xx, spl(xx)*(spl(xx)>0),k=1)
# dx_nu = nu.derivative(1)

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

def f_nonlinear(x,y,z):
    return 1/2*nu.antiderivative(1)(x**2+y**2+z**2)


def fx_nonlinear(x,y,z):
    return nu(x**2+y**2+z**2)*x

def fy_nonlinear(x,y,z):
    return nu(x**2+y**2+z**2)*y

def fz_nonlinear(x,y,z):
    return nu(x**2+y**2+z**2)*z


def fxx_nonlinear(x,y,z):
    return nu(x**2+y**2+z**2) + 2*x*dx_nu(x**2+y**2+z**2)*x

def fxy_nonlinear(x,y,z):
    return 2*x*y*dx_nu(x**2+y**2+z**2)

def fxz_nonlinear(x,y,z):
    return 2*x*z*dx_nu(x**2+y**2+z**2)


def fyx_nonlinear(x,y,z):
    return 2*x*y*dx_nu(x**2+y**2+z**2)

def fyy_nonlinear(x,y,z):
    return nu(x**2+y**2+z**2) + 2*y*dx_nu(x**2+y**2+z**2)*y

def fyz_nonlinear(x,y,z):
    return 2*y*z*dx_nu(x**2+y**2+z**2)


def fzx_nonlinear(x,y,z):
    return 2*x*z*dx_nu(x**2+y**2+z**2)

def fzy_nonlinear(x,y,z):
    return 2*y*z*dx_nu(x**2+y**2+z**2)

def fzz_nonlinear(x,y,z):
    return nu(x**2+y**2+z**2) + 2*z*dx_nu(x**2+y**2+z**2)*z




f_linear = lambda x,y,z : 1/2*nu0*(x**2+y**2+z**2)

fx_linear = lambda x,y,z : nu0*x
fy_linear = lambda x,y,z : nu0*y
fz_linear = lambda x,y,z : nu0*z

fxx_linear = lambda x,y,z : nu0 + 0*x
fxy_linear = lambda x,y,z : y*0
fxz_linear = lambda x,y,z : z*0

fyx_linear = lambda x,y,z : x*0
fyy_linear = lambda x,y,z : nu0 + 0*y
fyz_linear = lambda x,y,z : z*0

fzx_linear = lambda x,y,z : x*0
fzy_linear = lambda x,y,z : y*0
fzz_linear = lambda x,y,z : nu0 + z*0

##########################################################################################

# TODO:
    

def g_nonlinear(x,y,z):
    return 1/2*mu.antiderivative(1)(x**2+y**2+z**2)


def gx_nonlinear(x,y,z):
    return mu(x**2+y**2+z**2)*x

def gy_nonlinear(x,y,z):
    return mu(x**2+y**2+z**2)*y

def gz_nonlinear(x,y,z):
    return mu(x**2+y**2+z**2)*z


def gxx_nonlinear(x,y,z):
    return mu(x**2+y**2+z**2) + 2*x*dx_mu(x**2+y**2+z**2)*x

def gxy_nonlinear(x,y,z):
    return 2*x*y*dx_mu(x**2+y**2+z**2)

def gxz_nonlinear(x,y,z):
    return 2*x*z*dx_mu(x**2+y**2+z**2)


def gyx_nonlinear(x,y,z):
    return 2*x*y*dx_mu(x**2+y**2+z**2)

def gyy_nonlinear(x,y,z):
    return mu(x**2+y**2+z**2) + 2*y*dx_mu(x**2+y**2+z**2)*y

def gyz_nonlinear(x,y,z):
    return 2*y*z*dx_mu(x**2+y**2+z**2)


def gzx_nonlinear(x,y,z):
    return 2*x*z*dx_mu(x**2+y**2+z**2)

def gzy_nonlinear(x,y,z):
    return 2*y*z*dx_mu(x**2+y**2+z**2)

def gzz_nonlinear(x,y,z):
    return mu(x**2+y**2+z**2) + 2*z*dx_mu(x**2+y**2+z**2)*z


g_linear = lambda x,y,z : 1/(2*nu0)*(x**2+y**2+z**2)

gx_linear = lambda x,y,z : 1/nu0*x
gy_linear = lambda x,y,z : 1/nu0*y
gz_linear = lambda x,y,z : 1/nu0*z

gxx_linear = lambda x,y,z : 1/nu0 + 0*x
gxy_linear = lambda x,y,z : y*0
gxz_linear = lambda x,y,z : z*0

gyx_linear = lambda x,y,z : x*0
gyy_linear = lambda x,y,z : 1/nu0 + 0*y
gyz_linear = lambda x,y,z : z*0

gzx_linear = lambda x,y,z : x*0
gzy_linear = lambda x,y,z : y*0
gzz_linear = lambda x,y,z : 1/nu0 + z*0
           


# # evtl:
# a = -2.822*1e-10
# b = 2.529*1e-5
# c = 1.591
# Ms = 2.16
# mu0 = (4*np.pi)/10**7
