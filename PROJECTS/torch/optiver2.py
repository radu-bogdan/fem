import numpy as np

# Test case from question 1, as a check
f1 = lambda x,y : 1*(x<20)*(y<20)*(x>-20)*(y>-20)

# Test case from question 3
f2 = lambda x,y : 1*(((x-2.5)/30)**2+((y-2.5)/40)**2<1)

def go(point,f):
    
    length = 0
    
    while (1):
    
        d = np.random.randint(4);
        
        if d == 0: point = point + np.r_[ 10,   0]
        if d == 1: point = point + np.r_[-10,   0]
        if d == 2: point = point + np.r_[  0 ,-10]
        if d == 3: point = point + np.r_[  0 , 10]
        
        length = length + 1
        
        if f(*point)==0:
            return length

tries = 10_000 
cumsum = 0
start = np.r_[0,0]

for i in range(tries):
    cumsum = cumsum + go(start,f1)
print('Question 1, expected value approx: ', cumsum/tries)

cumsum = 0
for i in range(tries):
    cumsum = cumsum + go(start,f2)
print('Question 3, expected value approx: ', cumsum/tries)