import scipy as sp

def spxoradd_AB(A,B):
    A.eliminate_zeros()
    B.eliminate_zeros()    
    
    AB = A.multiply(B)
    AB.eliminate_zeros()
    # print(AB.data)
    AB.data = 1+0*AB.data
    
    # return A+B-A.multiply(AB)-B.multiply(AB)
    return (A+B)-(A+B).multiply(AB)

def spxoradd(*args):
    
    if len(args)<2: print('At least two args!')
    
    xorR = spxoradd_AB(args[0],args[1])
    
    for i in range(2,len(args)):
        xorR = spxoradd_AB(xorR,args[i])
        
    return xorR