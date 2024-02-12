import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')


J1 = lambda x,y,z : np.c_[ 1+0*x, 0*y, 0*z]
J2 = lambda x,y,z : np.c_[ 0*x, 1+0*y, 0*z]
J3 = lambda x,y,z : np.c_[-1+0*x, 0*y, 0*z]
J4 = lambda x,y,z : np.c_[ 0*x,-1+0*y, 0*z]


J = lambda x,y,z : np.c_[(x<75)*(x>-75),(y>-95)*(y<-75),1+z*0]*J1(x,y,z) +\
                   np.c_[(x<75)*(x>-75),(y> 75)*(y< 95),1+z*0]*J3(x,y,z)


J = lambda x,y,z : np.tile(((x<75)*(x>-75)*(y>-95)*(y<-75)),(3,1)).T*J1(x,y,z) +\
                   np.tile(((x<75)*(x>-75)*(y> 75)*(y< 95)),(3,1)).T*J3(x,y,z) +\
                   np.tile(((x<-75)*(x>-95)*(y<75)*(y>-75)),(3,1)).T*J4(x,y,z) +\
                   np.tile(((x> 75)*(x< 95)*(y<75)*(y>-75)),(3,1)).T*J2(x,y,z)
                   
                   
# J = lambda x,y,z : J1(x,y,z)
# J = lambda x,y,z : J2(x,y,z)


# Make the grid
x, y, z = np.meshgrid(np.arange(-150, 150, 10),
                      np.arange(-150, 150, 10),
                      np.arange( -70,  70, 10))

x = x.flatten()
y = y.flatten()
z = z.flatten()


# Make the direction data for the arrows
# u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
# v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
# w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
#      np.sin(np.pi * z))

u = J(x,y,z)[:,0]
v = J(x,y,z)[:,1]
w = J(x,y,z)[:,2]

ax.quiver(x, y, z, u, v, w, length=10, normalize=True)

plt.show()