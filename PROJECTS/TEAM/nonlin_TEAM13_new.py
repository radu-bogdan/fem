import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numba as nb

mu0 = (4*np.pi)/10**7
nu0 = 10**7/(4*np.pi)
mur = 1000

def mu(h):
    val = mu0*(2*mur-(2*mur-1)*(1/2+1/np.pi*np.arctan(h/3000)))
    if np.isnan(val):
        return mu0
    else:
        return val
    
mu = np.vectorize(mu)

xx = np.linspace(0,50,8000)
yy = np.exp(np.linspace(0,np.log(50*1e8),8000))-1



mu2 = interpolate.Akima1DInterpolator(yy, mu(yy**2))
dx_mu = mu2.derivative(1)

# series = yy*mu(yy)
# k = np.argsort(series)

nu = interpolate.Akima1DInterpolator(yy*mu(yy), (1/mu(yy)))
dx_nu = nu.derivative(1)



plt.close('all')
plt.figure()
plt.plot(yy,mu(yy),'*')
plt.figure()
plt.plot(xx,nu(xx),'*')

plt.figure()
plt.plot(mu(yy)*nu(mu(yy)*yy))
plt.plot(nu(xx)*mu(nu(xx)*xx))

# stop














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
    return 1/2*mu2.antiderivative(1)(x**2+y**2+z**2)


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
