import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)  # Disable bounds checking for extra speed
@cython.wraparound(False)  # Disable negative index wrapping
def sum_array(cnp.ndarray[cnp.float64_t, ndim=1] arr):
    cdef Py_ssize_t i  # Use C integer type for indices
    cdef double total = 0  # Use C double type for the accumulator

    # Loop through the array elements and sum them up
    for i in range(arr.shape[0]):
        total += arr[i]
    
    return total