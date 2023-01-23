import numpy as np


def x(t):
    return np.sinh(np.tan(t))

tau = 0.2
T = 1
xn = 0
tn = 0

for n in range(int(T/tau)):
    xn = xn + tau*(1/np.cos(tn)**2*np.sqrt(1+xn**2))
    tn = (n+1)*tau
    xtn = x(tn)
    relerror = abs(xn-xtn)/abs(xtn)
    print('time = ', "{:.2f}".format(tn),
          ', xn : ' "{:.8f}".format(xn),
          ', x(tn) : ' "{:.8f}".format(xtn),
          ', rel.error : ' "{:.8f}".format(relerror))