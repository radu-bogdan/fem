import numpy as np
import matplotlib.pyplot as plt

# from numpy.random import 

rng = np.random.default_rng()

# rng = default_rng()
# numbers = rng.choice(20, size=10, replace=False)

primes = np.r_[2,3,5,7,11,13,17,19,23,29,31,37,41,31,47,53]

# np.random.seed(312)

tries = 100_000_0
count = np.zeros(tries)
c = np.array(4)

times_even = 0

for i in range(tries):
    
    if (i%10_000 == 0) and (i>0):
        print('After ',i,'tries: ',times_even/i)
    
    a = rng.choice(16,size=4,replace=False)
    
    if (np.sum(primes[a]) % 2 == 0):
        times_even +=1
    
    if (i>0):
        count[i] = times_even/i
        
print(times_even/tries)
plt.plot(count[2000:])