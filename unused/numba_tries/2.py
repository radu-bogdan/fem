import fastMethods
import time
import numpy as np
import numba as nb

@nb.njit()
def generateRandom(zeilen,spalten):
    a = np.random.rand(zeilen,spalten)
    b = np.random.rand(zeilen,spalten)
    return a,b


zeilen = 40_000_000
spalten = 3

tm = time.time()
a,b = generateRandom(zeilen,spalten)
print(time.time()-tm)

tm = time.time()
x = fastMethods.stacking(a,b)
print(time.time()-tm)

tm = time.time()
y = fastMethods.stacking2(a,b)
print(time.time()-tm)

print(np.linalg.norm(x-y))

# tm = time.time()
# ident_loops(x)
# print(time.time()-tm)