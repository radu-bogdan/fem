import numpy as np
import matplotlib.pyplot as plt
n = 15
t, x = np.meshgrid(np.linspace(0,4,n),np.linspace(-3,3,n))
u = np.ones((n,n))
v = x-np.exp(-t)*(t+2)+2
z = np.sqrt(u*u+v*v)
u = u/z
v = v/z
plt.figure(figsize=(5,5))
plt.quiver(t,x,u,v,color=(.1,.1,.9),units='xy',scale=4)
plt.xlim(0,4)
plt.ylim(-3.5,3.5)
t = np.linspace(0,4,100)

x0 = 3

werte = np.linspace(-2,0,9)
for x0 in werte:
    c1 = x0 - 5./4+2
    x = c1*np.exp(t)+1./2*np.exp(-t)*t+5./4*np.exp(-t)-2
    plt.plot(t,x,color='black',linewidth=1.5)
    plt.xlabel('t')
    plt.ylabel('x')

# plt.draw()
# plt.savefig("filename", format='eps')
plt.show()