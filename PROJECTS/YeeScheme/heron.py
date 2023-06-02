import numpy as np

x = 1
a = 70000
theta = 0.5
k = 3

bitl = np.log(a)



x = 250

for _ in range(10):
    x = 1/k*((k-1)*x+a/(x**(k-1)))
    # x = x - (x-a/x)/(1+a/x**2)
    print('Error',x-a**(1/k))