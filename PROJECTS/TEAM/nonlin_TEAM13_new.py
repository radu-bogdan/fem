import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numba as nb

mu0 = (4*np.pi)/10**7
nu0 = 10**7/(4*np.pi)

a = -2.381*1e-10
b =  2.327*1e-5
c =  1.590

Ms = 2.16

f1 = lambda x : mu0*x  +a*x**2 + b*x + c
f2 = lambda x : mu0*x + Ms

B1 = np.r_[1.81:2.21:0.01]
B2 = np.r_[2.22:20:0.01]



H1 = (-b - mu0 + np.sqrt((b+mu0)**2-4*a*(c-B1)))/(2*a)
H2 = 1/mu0*(B2-Ms)

B = np.array([0, 0.0025, 0.0050, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80])
H = np.array([0, 16, 30, 54, 93, 143, 191, 210, 222, 233, 247, 258, 272, 289, 313, 342, 377, 433, 509, 648, 933, 1228, 1934, 2913, 4993, 7189, 9423])

B = np.r_[B,B1,B2]
H = np.r_[H,H1,H2]


# plt.plot(H,B,'*')
# plt.plot(H1,B1,'*')
# plt.plot(H2,B2,'*')


nu = interpolate.CubicSpline(B[1:]**2, H[1:]/B[1:], bc_type = 'natural')
mu = interpolate.CubicSpline(H[1:]**2, B[1:]/H[1:], bc_type = 'natural')

spl = interpolate.make_smoothing_spline(B[1:]**2, nu.derivative()(B[1:]**2))
dx_nu = interpolate.BSpline(B[1:]**2, spl(B[1:]**2)*(spl(B[1:]**2)>0),k=1)

spl = interpolate.make_smoothing_spline(H[1:]**2, nu.derivative()(H[1:]**2))
dx_mu = interpolate.BSpline(H[1:]**2, spl(H[1:]**2)*(spl(H[1:]**2)>0),k=1)


# dx_nu = nu.derivative(1)
# dx_mu = nu.derivative(1)


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
