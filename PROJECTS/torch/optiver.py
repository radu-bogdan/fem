import numpy as np


f = lambda x,y : 1*(x<=1)*(y<=1)*(x>=-1)*(y>=-1)

start = np.r_[0,0]

depth = 9

path_numbers   = np.zeros(depth)
path_completed = np.zeros(depth)


def go(point,length):
    
    if length == depth:
        return;
    
    # go right
    point = point + np.r_[1,0]
    length = length + 1
    
    if not is_it_in(point, f):
        path_completed[length] = path_completed[length] + 1
        return;
    
    go(point,length)
        
    # go left
    point = point + np.r_[-1,0]
    length = length + 1
    if not is_it_in(point, f):
        path_completed[length] = path_completed[length] + 1
        return;
    
    go(point,length)
        
    # go down
    point = point + np.r_[0,-1]
    length = length + 1
    if not is_it_in(point, f):
        path_completed[length] = path_completed[length] + 1
        return;
    
    go(point,length)
        
    # go up
    point = point + np.r_[0,1]
    length = length + 1
    if not is_it_in(point, f):
        path_completed[length] = path_completed[length] + 1
        return;
    
    go(point,length)
    
def is_it_in(point,f):
    if f(*point)==0:
        return 0
    else:
        return 1


a = go(start,0)

# print(f(-1,1))