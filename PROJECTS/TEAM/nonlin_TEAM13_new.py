import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numba as nb

mu0 = (4*np.pi)/10**7
nu0 = 10**7/(4*np.pi)

B0 = np.array([0, 0.0025, 0.0050, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80])
H0 = np.array([0, 16, 30, 54, 93, 143, 191, 210, 222, 233, 247, 258, 272, 289, 313, 342, 377, 433, 509, 648, 933, 1228, 1934, 2913, 4993, 7189, 9423])

a = -2.381*1e-10
b =  2.327*1e-5
c =  1.590

Ms = 2.16

f1 = lambda x : mu0*x  +a*x**2 + b*x + c
f2 = lambda x : mu0*x + Ms

B1 = np.r_[1.81:2.21:0.01]
B2 = np.r_[2.22:200:0.01]

H1 = (-b - mu0 + np.sqrt((b+mu0)**2-4*a*(c-B1)))/(2*a)
H2 = 1/mu0*(B2-Ms)

B = np.r_[B0,B1,B2]
H = np.r_[H0,H1,H2]

# plt.plot(H,B,'*')



gss_vals = (B[1:]-B[:-1])/(H[1:]-H[:-1])
fss_vals = 1/gss_vals

# plt.figure()
# plt.plot(mu,'*')

# plt.plot(H[:-1], mu_vals, '*')
# plt.plot(B[:-1], nu_vals, '*')

gss = interpolate.CubicSpline(H[:-1], gss_vals, bc_type = 'natural')
fss = interpolate.CubicSpline(B[:-1], fss_vals, bc_type = 'natural')

gs = gss.antiderivative(1)
fs = fss.antiderivative(1)

g = gs.antiderivative(1)
f = fs.antiderivative(1)

# plt.plot(H[:-1], gs(H[:-1]), '*')
# plt.plot(B[:-1], fss(B[:-1]), '*')



# def mu(h):
#     val = mu0*(2*mur-(2*mur-1)*(1/2+1/np.pi*np.arctan(h/3000)))
#     if np.isnan(val):
#         return mu0
#     else:
#         return val
    
# mu = np.vectorize(mu)

# xx = np.linspace(0,50,8000)
# yy = np.exp(np.linspace(0,np.log(50*1e8),8000))-1



# mu2 = interpolate.Akima1DInterpolator(yy, mu(yy**2))
# dx_mu = mu2.derivative(1)

# # series = yy*mu(yy)
# # k = np.argsort(series)

# nu = interpolate.Akima1DInterpolator(yy*mu(yy), (1/mu(yy)))
# dx_nu = nu.derivative(1)



# plt.close('all')
# plt.figure()
# plt.plot(yy,mu(yy),'*')
# plt.figure()
# plt.plot(xx,nu(xx),'*')

# plt.figure()
# plt.plot(mu(yy)*nu(mu(yy)*yy))
# plt.plot(nu(xx)*mu(nu(xx)*xx))

# # stop








eps = 1e-5


def norms(x,y,z):
    val = (x**2+y**2+z**2+eps)**0.5
    # return val*(val>eps) + eps*(val<eps)
    return val

eps = 1e-5

def f_nonlinear(x,y,z):
    return f(norms(x,y,z))

def fx_nonlinear(x,y,z):
    return fs(norms(x,y,z))*x/norms(x,y,z)

def fy_nonlinear(x,y,z):
    return fs(norms(x,y,z))*y/norms(x,y,z)

def fz_nonlinear(x,y,z):
    return fs(norms(x,y,z))*z/norms(x,y,z)


def fxx_nonlinear(x,y,z):
    return fss(norms(x,y,z))*x*x/norms(x,y,z)**2 + fs(norms(x,y,z))*(y**2+z**2)/norms(x,y,z)**3 + eps
    
def fxy_nonlinear(x,y,z):
    return fss(norms(x,y,z))*x*y/norms(x,y,z)**2 + fs(norms(x,y,z))*(-x*y)/norms(x,y,z)**3

def fxz_nonlinear(x,y,z):
    return fss(norms(x,y,z))*x*z/norms(x,y,z)**2 + fs(norms(x,y,z))*(-x*z)/norms(x,y,z)**3


def fyx_nonlinear(x,y,z):
    return fss(norms(x,y,z))*x*y/norms(x,y,z)**2 + fs(norms(x,y,z))*(-x*y)/norms(x,y,z)**3

def fyy_nonlinear(x,y,z):
    return fss(norms(x,y,z))*y*y/norms(x,y,z)**2 + fs(norms(x,y,z))*(x**2+z**2)/norms(x,y,z)**3 + eps

def fyz_nonlinear(x,y,z):
    return fss(norms(x,y,z))*y*z/norms(x,y,z)**2 + fs(norms(x,y,z))*(-y*z)/norms(x,y,z)**3


def fzx_nonlinear(x,y,z):
    return fss(norms(x,y,z))*x*z/norms(x,y,z)**2 + fs(norms(x,y,z))*(-x*z)/norms(x,y,z)**3

def fzy_nonlinear(x,y,z):
    return fss(norms(x,y,z))*y*z/norms(x,y,z)**2 + fs(norms(x,y,z))*(-y*z)/norms(x,y,z)**3

def fzz_nonlinear(x,y,z):
    return fss(norms(x,y,z))*z*z/norms(x,y,z)**2 + fs(norms(x,y,z))*(x**2+y**2)/norms(x,y,z)**3 + eps




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

# ##########################################################################################

# # TODO:
    

def g_nonlinear(x,y,z):
    return g(norms(x,y,z))

def gx_nonlinear(x,y,z):
    return gs(norms(x,y,z))*x/norms(x,y,z)

def gy_nonlinear(x,y,z):
    return gs(norms(x,y,z))*y/norms(x,y,z)

def gz_nonlinear(x,y,z):
    return gs(norms(x,y,z))*z/norms(x,y,z)


def gxx_nonlinear(x,y,z):
    return gss(norms(x,y,z))*x*x/norms(x,y,z)**2 + gs(norms(x,y,z))*(y**2+z**2)/norms(x,y,z)**3 + eps
    
def gxy_nonlinear(x,y,z):
    return gss(norms(x,y,z))*x*y/norms(x,y,z)**2 + gs(norms(x,y,z))*(-x*y)/norms(x,y,z)**3

def gxz_nonlinear(x,y,z):
    return gss(norms(x,y,z))*x*z/norms(x,y,z)**2 + gs(norms(x,y,z))*(-x*z)/norms(x,y,z)**3


def gyx_nonlinear(x,y,z):
    return gss(norms(x,y,z))*x*y/norms(x,y,z)**2 + gs(norms(x,y,z))*(-x*y)/norms(x,y,z)**3

def gyy_nonlinear(x,y,z):
    return gss(norms(x,y,z))*y*y/norms(x,y,z)**2 + gs(norms(x,y,z))*(x**2+z**2)/norms(x,y,z)**3 + eps

def gyz_nonlinear(x,y,z):
    return gss(norms(x,y,z))*y*z/norms(x,y,z)**2 + gs(norms(x,y,z))*(-y*z)/norms(x,y,z)**3


def gzx_nonlinear(x,y,z):
    return gss(norms(x,y,z))*x*z/norms(x,y,z)**2 + gs(norms(x,y,z))*(-x*z)/norms(x,y,z)**3

def gzy_nonlinear(x,y,z):
    return gss(norms(x,y,z))*y*z/norms(x,y,z)**2 + gs(norms(x,y,z))*(-y*z)/norms(x,y,z)**3

def gzz_nonlinear(x,y,z):
    return gss(norms(x,y,z))*z*z/norms(x,y,z)**2 + gs(norms(x,y,z))*(x**2+y**2)/norms(x,y,z)**3 + eps


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
