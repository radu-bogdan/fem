import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numba as nb

#
# Annahme : H = nu(|B|**2) B
#           B = mu(|H|**2) H
# 


nu0 = 10**7/(4*np.pi)
k1 = 49.4; k2 = 1.46; k3 = 520.6


@nb.njit()
def nu_nl(x):
    return (k1*np.exp(k2*(x))+k3)
    

@nb.njit()
def nu(x):
    pos = 1/k2*np.log(nu0/(k1*k2))
    m = pos-1/nu0*nu_nl(pos)
    return (k1*np.exp(k2*(x))+k3)*(x<pos)+nu0*(x-m)*(x>pos)


B_KL = np.exp(np.linspace(0,3,100))-1
# B_KL = np.linspace(0,3,1000)
H_KL = nu(B_KL)
H_KL = H_KL.astype(float)


HB_curve = interpolate.CubicSpline(H_KL, B_KL, bc_type='natural')
coenergy_density = HB_curve.antiderivative()
dx_HB_curve = HB_curve.derivative()

BH_curve = interpolate.CubicSpline(B_KL, H_KL, bc_type='natural')
energy_density = BH_curve.antiderivative()
dx_BH_curve = BH_curve.derivative()

ab = 1
nu_curve = interpolate.CubicSpline(B_KL[ab:]**2, H_KL[ab:]/B_KL[ab:], bc_type=[[2,0],[2,0]], extrapolate=True)
dx_nu_curve = nu_curve.derivative()

mu_curve = interpolate.CubicSpline(H_KL[ab:]**2, B_KL[ab:]/H_KL[ab:], bc_type=[[2,0],[2,0]], extrapolate=True)
dx_mu_curve = nu_curve.derivative()



# mu_curve = lambda x : HB_curve(x)/(x+1e-12)
# nu_curve = lambda x : BH_curve(x)/(x+1e-12)

# dx_mu_curve = lambda x : (dx_HB_curve(x)*x-HB_curve(x))/(x**2)
# dx_nu_curve = lambda x : (dx_BH_curve(x)*x-BH_curve(x))/(x**2)


# plt.figure()
# plt.plot()

# plt.plot(H_KL,coenergy_density(H_KL))


xx = np.linspace(0,10,1_000_000)

plt.figure()
# plt.plot(xx,HB_curve(xx))
plt.plot(xx,dx_BH_curve(xx))
# plt.plot(xx,nu(xx))
# plt.plot(xx,nu0_lin(xx))
# plt.plot(xx,BH_curve(xx))
# plt.plot(xx,BH_curve.derivative(2)(xx))
# plt.plot(B_KL,H_KL,'.')








# def nu(x,y):
    # k1 = 49.4; k2 = 1.46; k3 = 520.6
#     return k1*np.exp(k2*(x**2+y**2))+k3


# plt.plot(xx,nu(xx**(0.5),xx**(0.5)))




