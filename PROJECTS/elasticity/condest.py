import numpy as np

def condest(A):
    t = 2
    itmax = 5
    it = 0
    
    sA = A.shape[0]
    
    x = np.ones((sA,t))
    
    
    
    while True:
        it = it + 1
        
        


import scipy.sparse as sp

A = sp.random(m = 100, n = 100, density = 0.01, format = 'csc', dtype = np.float64)

condest(A)