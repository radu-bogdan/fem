import numpy as np
import matplotlib.pyplot as plt

lam = 0.118613686396592868190663397504305416107192735394247300219882
bet = 0.307745941625991646104616284246250960038293608432243697986116
gam = 0.149319748020505695949137692011715113502703657706172951310642

we1 = 0.003565179653602241016812012063893264122299237666447620688443
we2 = 0.014784708088402646966377737832150897944929423471101953572709
we3 = 0.050942326513475907075701900282153205791750927286762352959923
we4 = 0.082589744322783224641397278656318400862757654771252785872881

qp = np.c_[[0,   0],
           [bet, 0],
            [1-bet, 0],
            [1,   0],
            [bet, 1-bet],
            [1-bet, bet],
            [0, 1],
            [0, bet],
            [0, 1-bet],
            [lam, lam],
            [lam, 1-2*lam],
            [1-2*lam, lam],
            [1/2-1/2*gam, 1/2-1/2*gam],
            [gam, 1/2-1/2*gam],
            [1/2-1/2*gam, gam]]

we = 2*np.r_[we1,we2,we2,we1,we2,we2,we1,we2,we2,we3,we3,we3,we4,we4,we4]

# plt.plot(qp[0,:],qp[1,:],'.')

p1 = np.r_[0,0]
p2 = np.r_[1,0]
p3 = np.r_[1/2,np.sqrt(3)/2]

F = lambda x,y : np.c_[p2-p1,p3-p1] @ np.c_[x,y].T + np.tile(p1,(x.size,1)).T

qp2 = F(qp[0,:].T,qp[1,:].T)

plt.plot(qp2[0,:],qp2[1,:],'.')

def f(fun):
    summe = 0;
    for i,j in enumerate(qp.T):
        print(i,j)
        summe = summe + 1/2*we[i]*fun(j[0],j[1])
    return summe

print(f(lambda x,y: y**3*x**2))
print('sympy shit:')



# from sympy.integrals.intpoly import *
# a = polytope_integrate(Polygon((0, 0), (0, 1), (1, 0)), x*(1-x-y) )
# print(a)
# print(float(a))
