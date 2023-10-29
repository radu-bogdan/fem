import numpy as np
import matplotlib.pyplot as plt

n = 20
t, x = np.meshgrid(np.linspace(-3,3,n),np.linspace(-3,3,n))
u = np.ones((n,n))

# v = x-np.exp(-t)*(t+2)+2
v = -2*x+3/t
# v = 1/x*(3+np.tan(x))



z = np.sqrt(u*u+v*v)
u = u/z
v = v/z
plt.figure(figsize=(5,5))
plt.quiver(t,x,u,v,color=(.1,.1,.9),units='xy',scale=4)
plt.xlim(-3,3)
plt.ylim(-3,3)
# t = np.linspace(0,4,100)

# x0 = 3

# werte = np.linspace(-2,0,9)
# for x0 in werte:
#     c1 = x0 - 5./4+2
#     x = c1*np.exp(t)+1./2*np.exp(-t)*t+5./4*np.exp(-t)-2
       
#     plt.plot(t,x,color='black',linewidth=1.5)
#     plt.xlabel('t')
#     plt.ylabel('x')

# plt.draw()
# plt.savefig("filename", format='eps')
plt.show()


f = lambda h : (1+h)**(1/h)*np.exp(1)