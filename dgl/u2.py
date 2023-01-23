import numpy as np
import matplotlib.pyplot as plt

tmin = 0
tmax = 1.6
xmin = -3
xmax = 3

n = 16+1
t, x = np.meshgrid(np.linspace(tmin,tmax,n),np.linspace(xmin,xmax,n))
u = np.ones((n,n))
v = (2*x-3)*(3*t-np.sin(2*t))
z = np.sqrt(u*u+v*v)
u = u/z
v = v/z
plt.figure(figsize=(5,5))
plt.quiver(t,x,u,v,color=(0.1,0.1,0.9),units='xy',scale=4, width=0.03)
plt.xlim(tmin,tmax)
plt.ylim(xmin,xmax)
t = np.linspace(tmin,tmax,100)

werte = np.linspace(3/2,3/2,1)
for x0 in werte:
    c1 = (x0-3/2)*np.exp(-1)
    x = c1*np.exp(3*t**2+np.cos(t))+3/2
    plt.plot(t,x,color='black',linewidth=1.5)
    plt.xlabel('t')
    plt.ylabel('x')

plt.draw()
plt.savefig("eehg", format='eps')
plt.show()