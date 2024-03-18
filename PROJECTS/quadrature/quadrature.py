import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

p1 = np.r_[0,0]
p2 = np.r_[1,0]
p3 = np.r_[1/2,np.sqrt(3)/2]

m1 = (p2+p3)/2
m2 = (p1+p3)/2
m3 = (p1+p2)/2

b = (p1+p2+p3)/3

T1 = np.c_[p1,p2,p3] # vertices
T2 = np.c_[m1,m2,m3] # edge midpoints
T3 = b   # midpoint

T4 = lambda a : np.c_[a*p1 + (1-a)*p2,
                      a*p2 + (1-a)*p1,
                      a*p3 + (1-a)*p1,
                      a*p1 + (1-a)*p3,
                      a*p3 + (1-a)*p2,
                      a*p2 + (1-a)*p3,] # edge class

T5 = lambda a : np.c_[a*m1 + (1-a)*p1,
                      a*m2 + (1-a)*p2,
                      a*m3 + (1-a)*p3] # inner class, type 1

T6 = lambda a,b : np.c_[b*(a*m1 + (1-a)*p1) + (1-b)*(a*m2 + (1-a)*p2),
                        b*(a*m1 + (1-a)*p1) + (1-b)*(a*m3 + (1-a)*p3),
                        b*(a*m3 + (1-a)*p3) + (1-b)*(a*m2 + (1-a)*p2),
                        b*(a*m2 + (1-a)*p2) + (1-b)*(a*m1 + (1-a)*p1),
                        b*(a*m3 + (1-a)*p3) + (1-b)*(a*m1 + (1-a)*p1),
                        b*(a*m2 + (1-a)*p2) + (1-b)*(a*m3 + (1-a)*p3)] # inner class, type 2

##### Polynomials #####

p = 6

def pol(p,x,y):
    res = np.empty(0)
    for i in range(p+1):
        for j in range(p+1):
            if i+j>=p+1:
                continue
            print(i,j,i+j,p+1,"x**%d+y**%d" %(i,j))
            # print(x**i+y**j)
            # print(x**i*y**j)
            res = np.r_[res, x**i*y**j]
    return res

##### Matrix assembly #####



# print(pol(2,0.1,0.4))
print(pol(2,np.c_[0.1,0.2,0.6],np.c_[0.4,0.6,0.7]))




P = lambda x,y : np.r_[1,x,y,x*y,x**2,y**2]







plt.plot(T1[0,:],T1[1,:],'.')
# plt.plot(T2[0,:],T2[1,:],'.')
# plt.plot(T3[0],T3[1],'.')
# a = 0.1
# plt.plot(T4(a)[0,:],T4(a)[1,:],'*')
a = 0.3
b = 0.3
plt.plot(T6(a,b)[0,:],T6(a,b)[1,:],'*')
plt.plot(T5(a)[0,:],T5(a)[1,:],'*')





# T1 = np.r_[]