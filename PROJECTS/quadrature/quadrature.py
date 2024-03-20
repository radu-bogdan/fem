import numpy as np
import matplotlib.pyplot as plt
import math
from sksparse.cholmod import cholesky as chol

plt.close('all')

p1 = np.r_[0,0]
p2 = np.r_[1,0]
# p3 = np.r_[1/2,np.sqrt(3)/2]
p3 = np.r_[0,1]

m1 = (p2+p3)/2
m2 = (p1+p3)/2
m3 = (p1+p2)/2

b = (p1+p2+p3)/3

# vertices
T1 = np.c_[p1,p2,p3]
eval_T1 = lambda i,j : T1[0,:]**i*T1[1,:]**j

# edge midpoints
T2 = np.c_[m1,m2,m3]
eval_T2 = lambda i,j : T2[0,:]**i*T2[1,:]**j

# midpoint
T3 = b   
eval_T3 = lambda i,j : T3[0,:]**i*T3[1,:]**j

# edge class
T4 = lambda a : np.c_[a*p1 + (1-a)*p2,
                      a*p2 + (1-a)*p1,
                      a*p3 + (1-a)*p1,
                      a*p1 + (1-a)*p3,
                      a*p3 + (1-a)*p2,
                      a*p2 + (1-a)*p3]
eval_T4 = lambda a,i,j : T4(a)[0,:]**(0*int(i<0)+i*int(i>=0))*T4(a)[1,:]**(0*int(j<0)+j*int(j>=0))

dT4 = np.c_[p1-p2,
            p2-p1,
            p3-p1,
            p1-p3,
            p3-p2,
            p2-p3]

eval_dT4 = lambda i,j : dT4[0,:]**(0*int(i<0)+i*int(i>=0))*dT4[1,:]**(0*int(j<0)+j*int(j>=0))

# inner class, type 1, TODO
T5 = lambda a : np.c_[a*m1 + (1-a)*p1,
                      a*m2 + (1-a)*p2,
                      a*m3 + (1-a)*p3]

dT5 = np.c_[m1-p1,
            m2-p2,
            m3-p3]

eval_T5 = lambda a,i,j : T5(a)[0,:]**(0*int(i<0)+i*int(i>=0))*T5(a)[1,:]**(0*int(j<0)+j*int(j>=0))

# inner class, type 2, TODO
T6 = lambda a,b : np.c_[b*(a*m1 + (1-a)*p1) + (1-b)*(a*m2 + (1-a)*p2),
                        b*(a*m1 + (1-a)*p1) + (1-b)*(a*m3 + (1-a)*p3),
                        b*(a*m3 + (1-a)*p3) + (1-b)*(a*m2 + (1-a)*p2),
                        b*(a*m2 + (1-a)*p2) + (1-b)*(a*m1 + (1-a)*p1),
                        b*(a*m3 + (1-a)*p3) + (1-b)*(a*m1 + (1-a)*p1),
                        b*(a*m2 + (1-a)*p2) + (1-b)*(a*m3 + (1-a)*p3)] 

##### Evaluation #####

def integral(i,j):
    return math.factorial(j)*math.factorial(i)/math.factorial(i+j+2)

###### System ######

Jij = lambda i,j,a,w : np.r_[sum(eval_T1(i,j)),
                             sum(eval_T2(i,j)),
                             sum(eval_T5(a,i,j)),
                             w[-1]*sum(eval_T5(a,i-1,j)*dT5[0,:] + eval_T5(a,i,j-1)*dT5[1,:])]

# res = lambda i,j,a,w : np.r_[eval_T1(i,j),eval_T2(i,j),eval_T5(a,i,j)]*w-integral(i,j)


lam = 2*(7-np.sqrt(13))/18;

we1 = (11-np.sqrt(13))/360;
we2 = (10-2*np.sqrt(13))/45;
we3 = (29+17*np.sqrt(13))/360;
J = lambda a,w : np.c_[Jij(0,0,a,w),
                       Jij(1,0,a,w),
                        Jij(2,0,a,w),
                        # Jij(0,1,a,w),
                        # Jij(1,1,a,w),
                        # Jij(0,2,a,w),
                       ]
rhs = lambda a,w : np.c_[1/2*Jij(0,0,a,w)[:-1]@w - integral(0,0),
                         1/2*Jij(1,0,a,w)[:-1]@w - integral(1,0),
                         # 1/2*Jij(1,1,a,w)[:-1]@w - integral(1,1),
                         1/2*Jij(2,0,a,w)[:-1]@w - integral(2,0)]


# given a,w : 
    

w = np.r_[1/3,1/3,1/3]
a = 1/3

for i in range(100):
    Jaw = J(a,w).T
    rhsaw = rhs(a,w)
    JS = Jaw.T@Jaw
    
    x = np.linalg.solve(Jaw.T@Jaw,Jaw.T@rhsaw.T)[:,0]
    
    w = w + 1/2*x[:3]
    a = a + 1/2*x[-1]
    
    print('wtf ',i,x,w,a)
    # print('wtf ',x[:3],x[-1])
    

# Jaw = J(a,w)
# rhsaw = rhs(a,w)
    
# JS = Jaw.T@Jaw
# rhsaw

# x = np.linalg.solve(JS,rhsaw.T)

# print(rhsaw)

# print(Jaw.shape)
# print((Jaw.T@Jaw).shape)
# print(np.linalg.matrix_rank(Jaw))

stop

##### Polynomials #####

p = 6



def pol(p,x,y):
    res = np.empty(0)
    res = []
    for i in range(p+1):
        for j in range(p+1):
            if i+j>=p+1:
                continue
            # print(i,j,i+j,p+1,"x**%d+y**%d" %(i,j))
            res = np.r_[res, x**i*y**j]
            # res.append(x**i*y**j)
    return res




import sympy as sym
x, y = sym.symbols('x y')
print(pol(2,x,y))

pol = x*y


T1[0,:]**2 + T1[1,:]**2



##### Matrix assembly #####



# print(pol(2,0.1,0.4))
# print(pol(2,np.c_[0.1,0.2,0.6],np.c_[0.4,0.6,0.7]))




P = lambda x,y : np.r_[1,x,y,x*y,x**2,y**2]







plt.plot(T1[0,:],T1[1,:],'.')
# plt.plot(T2[0,:],T2[1,:],'.')
# plt.plot(T3[0],T3[1],'.')
# a = 0.1
# plt.plot(T4(a)[0,:],T4(a)[1,:],'*')
a = 0.3
b = 0.3
# plt.plot(T6(a,b)[0,:],T6(a,b)[1,:],'*')
plt.plot(T5(a)[0,:],T5(a)[1,:],'*')





# T1 = np.r_[]