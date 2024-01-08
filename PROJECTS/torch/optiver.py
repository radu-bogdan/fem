# import numpy as np
import math
import time

# f = lambda x,y : 1*(x<=1)*(y<=1)*(x>=-1)*(y>=-1)

# start = np.r_[0,0]

# depth = 9

# path_numbers   = np.zeros(depth)
# path_completed = np.zeros(depth)


# def go(point,length):
    
#     if length == depth:
#         return;
    
#     # go right
#     point = point + np.r_[1,0]
#     length = length + 1
    
#     if not is_it_in(point, f):
#         path_completed[length] = path_completed[length] + 1
#         return;
    
#     go(point,length)
        
#     # go left
#     point = point + np.r_[-1,0]
#     length = length + 1
#     if not is_it_in(point, f):
#         path_completed[length] = path_completed[length] + 1
#         return;
    
#     go(point,length)
        
#     # go down
#     point = point + np.r_[0,-1]
#     length = length + 1
#     if not is_it_in(point, f):
#         path_completed[length] = path_completed[length] + 1
#         return;
    
#     go(point,length)
        
#     # go up
#     point = point + np.r_[0,1]
#     length = length + 1
#     if not is_it_in(point, f):
#         path_completed[length] = path_completed[length] + 1
#         return;
    
#     go(point,length)
    
# def is_it_in(point,f):
#     if f(*point)==0:
#         return 0
#     else:
#         return 1


# a = go(start,0)

# # print(f(-1,1))





# def primeCount(n):
#     prod = 1
#     count = 0
#     for i in range(2,int(1e10)):
        
#         isPrime = 1
        
#         #Check if i is prime:
#         for j in range(2,int(i**0.5)+1):
#             if (i%j==0): isPrime = 0
        
#         # if isPrime: print(i)
        
#         if isPrime : prod = prod * i
        
#         # print('|%d|%d|%d|%d|' %(i,n,count,prod))
        
#         if (prod>n):
#             break
        
#         if isPrime : count = count + 1
        
        
        
#         # if ((n%i==0) and isPrime):
#         #     print('For %d, %d' % (n,i))
#         #     count = count + 1
    
#     return int(count)


# print(primeCount(1))
# print(primeCount(2))
# print(primeCount(3))
# print(primeCount(500))
# print(primeCount(5000))
# print(primeCount(10000000000))





# def querSum(m):
#     s = 0
#     while (m>0):
#         s = s + (m%10)
#         m = m//10
#     return s

# if __name__ == '__main__':
#     n = 100000
    
#     c = 0
#     qsc = 0
    
#     for i in range(1,n+1):
#         if (n%i==0):
#             qs = querSum(i)
            
#             print('|%d|%d|%d|%d|' %(i,qs,qsc,c))
#             if ((qs>qsc) or ((qs==qsc) and (i>c))):
#                 c = i
#                 qsc = qs
#     print(c)
    
    
    
# def strangeGrid(r, c):
#     return (r%2==1)*(2*(c-1)+10*(r-1)//2)+(r%2==0)*(2*c-1+10*(r-1)//2)

# print(strangeGrid(6,3))



def divisors(x):
    xi = x
    pow2 = 0
    
    if (x%2!=0): return 0
    
    # print('x is %d' %(x))
    while (x%2==0) and (x>1):
        pow2 = pow2 + 1
        x = x//2
    
    # if pow2==0: return [0,0]
    
    # print('x is after %d' %(x))
    
    count = 0
   
    # print('x=%d and x after div two=%d'%(xi,x))
    
    # print(x,int(math.sqrt(x))+2)
    
    # print(x,int(x**0.5)+6)
    prod = 1
    xold = x
    for i in range(3,x+1,2):
    # for i in range(3,int(x**0.5),2):
    # for i in range(3,x+1,2):
        count = 0
        while (x%i==0) and (x>1):
            count = count + 1
            x = x//i
        # print('x is %d and i is %d, count is %d' %(x,i,count))
        
        prod = prod * (count+1)
        if (x<=1): break
    
    # prod = prod * 2
    # if prod == 1: prod = prod*2
    
    # print(xi,pow2,count,prod,'\n')
    return pow2*prod

def divisors2(n):
    count = 0
    
    if (n%2==1): return 0
    
    for i in range(1,int(n**0.5)+2):
        if (n%i==0) and (i%2==0):
            count = count + 1
    return count




xx = [i for i in range(100)]
# xx = [10]
# xx = [158260522,
# 877914575,
# 602436426,
# 24979445,
# 861648772,
# 623690081,
# 433933447,
# 476190629,
# 262703497,
# 211047202,
# 971407775,
# 628894325,
# 731963982,
# 822804784,
# 450968417,
# 430302156,
# 982631932,
# 161735902,
# 880895728,
# 923078537,
# 707723857,
# 189330739,
# 910286918,
# 802329211,
# 404539679,
# 303238506,
# 317063340,
# 492686568,
# 773361868,
# 125660016,
# 650287940,
# 839296263,
# 462224593,
# 492601449,
# 384836991,
# 191890310,
# 576823355,
# 782177068,
# 404011431,
# 818008580,
# 954291757,
# 160449218,
# 155374934,
# 840594328,
# 164163676,
# 797829355,
# 138996221,
# 501899080,
# 353195922,
# 545531545,
# 910748511,
# 350034067,
# 913575467,
# 470338674,
# 824284691,
# 533206504,
# 180999835,
# 31262034,
# 138344965,
# 677959980,
# 131381221,
# 846045895,
# 208032501,
# 346948152,
# 973708325,
# 506147731,
# 893229302,
# 816248153,
# 298309896,
# 37119022,
# 455797489,
# 215208399,
# 190870607,
# 189766466,
# 554374432,
# 137831502,
# 694053968,
# 47628333,
# 469187475,
# 409233792,
# 810900084,
# 888987010,
# 799979694,
# 87027328,
# 575207354,
# 421624712,
# 666538409,
# 871216480,
# 597902643,
# 509928121,
# 727751951,
# 14723492,
# 265804420,
# 569328197,
# 234828607,
# 807497925,
# 360472167,
# 537560717,
# 613363438,
# 889089200,
# ]
# print([x for x in xx])



tm = time.monotonic()
[divisors(x) for x in xx]
print([divisors(x) for x in xx])
print('Took %.2f sec.' %(time.monotonic()-tm))

tm = time.monotonic()
[divisors2(x) for x in xx]
print([divisors2(x) for x in xx])
print('Took %.2f sec.' %(time.monotonic()-tm))

print('\n\n')
print(sum([divisors2(x)-divisors(x) for x in xx]))
print([divisors2(x)-divisors(x) for x in xx])

