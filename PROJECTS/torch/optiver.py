import numpy as np


f = lambda x,y : 1*(x<=1)*(y<=1)*(x>=-1)*(y>=-1)

start = np.r_[0,0]

depth = 10


def go(point,length,cumsum):
    
    # go right
    point = point + np.r_[1,0]
    length = length + 1
    if not is_it_in(point, f):
        cumsum = cumsum + length
        return;
        
    # go left
    point = point + np.r_[-1,0]
    length = length + 1
    if not is_it_in(point, f):
        cumsum = cumsum + length
        return;
        
    # go down
    point = point + np.r_[0,-1]
    length = length + 1
    if not is_it_in(point, f):
        cumsum = cumsum + length
        return;
        
    # go up
    point = point + np.r_[0,1]
    length = length + 1
    if not is_it_in(point, f):
        cumsum = cumsum + length
        return;
    
def is_it_in(point,f):
    if f(*point)==0:
        return 0
    else:
        return 1


a = go(start,0,0)

# print(f(-1,1))