import numpy as np
import time
import matplotlib.pyplot as plt


def mergeSort(arr):
    
    size = arr.size // 2
    # print('arrsize',size)
    
    if arr.size == 1:
        return arr
    
    arrL = mergeSort(arr[:size])
    arrR = mergeSort(arr[size:])
    
    # hier sind die arrL und arrR bereits .. gesortet?    
    
    arrLs = arrL.size
    arrRs = arrR.size
    
    # print(arrL,arrR)
    
    mergedArr = np.zeros(arrLs + arrRs)
    
    i = 0; j = 0;
    # print('ai jay',i,j)
    for k in range(arrLs + arrRs):
        if j == arrRs:
            mergedArr[k:] = arrL[i:]
            return mergedArr
        if i == arrLs:
            mergedArr[k:] = arrR[j:]
            return mergedArr
        
        if arrL[i]<arrR[j]:
            mergedArr[k] = arrL[i]
            i = i + 1
        else:
            mergedArr[k] = arrR[j]
            j = j + 1
            
    return mergedArr

def bubbleSort(arr):
    for _ in range(arr.size):
        swapped = False
        for i in range(arr.size-1):
            if arr[i+1]<arr[i]:
                s = arr[i+1]
                arr[i+1] = arr[i]
                arr[i] = s
                swapped = True
        
        if swapped == False:
            break
    
    return arr



def convergence(algo,it):
    times = np.zeros(it)
    sizes = np.zeros(it)
    N = 10
    for i in range(it):
        N = N*2
        arr = np.random.randint(0,N**2,N)
        tm = time.monotonic()
        algo(arr)
        times[i] = time.monotonic()-tm
        sizes[i] = N
        print(times)
    return sizes,times



N = 11;
arr = np.random.randint(0,N**2,N)

# arr = np.array([ 58, 114, 102,   9,  11, 119,  47,  32,  30, 118, 101])
# print(arr)
# print(mergeSort(arr))



sizes, times = convergence(bubbleSort,10)
plt.loglog(sizes, times,'*r')
plt.loglog(*convergence(mergeSort,10),'+')
plt.loglog(*convergence(np.sort,10),'--k')
plt.loglog(sizes,sizes**2,)
plt.loglog(sizes,sizes*np.log(sizes))
# plt.loglog(sizes,sizes,'--')


# swap two values without third variable? wie ging des nomma?