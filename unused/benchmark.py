import time
import numpy as np
import scipy as sp
import scipy.linalg

t = time.time() # do stuff 
A = np.random.rand(1000,1000)
B = np.random.rand(1000,1000)
# e1 = time.time() - t

# t = time.time() 
# C = A@B
# e2 = time.time() - t


# t = time.time() 
# C = np.linalg.solve(A, B)
# e3 = time.time() - t

# print('Initializing took {:.2f} seconds, multiplying took {:.2f} seconds, solving took {:.2f}'.format(e1,e2,e3))


P, L, U = scipy.linalg.lu(A)

D = P@L@U-A

print(scipy.linalg.norm(D, np.inf))