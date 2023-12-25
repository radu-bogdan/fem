
import numpy as np
import math

# @profile
def one_d(order):
    points = lambda x : np.linspace(0,1,x)
    if order == 0:
        qp = np.r_[1/2]
        we = np.array([1])
    if order == 1:
        qp = points(2)
        we = np.array([1/2,1/2])
    if order == 2:
        qp = np.array([1/2-np.sqrt(3)/6,1/2+np.sqrt(3)/6])# Gauss, second order
        we = np.array([1/2,1/2])
        # qp = np.r_[1/2-np.sqrt(3)/6,1/2+np.sqrt(3)/6] 
        # we = 1/2*np.r_[1,1]
    if order == 3:
        qp = points(3)
        # we = 1/6*np.r_[1,4,1]
        we = np.array([1/6,2/3,1/6])
    if order == 3:
        qp = points(4)
        # we = 1/8*np.r_[1,3,3,1]
        we = np.array([1/8,3/8,3/8,1/8])
    if (order == 5) or (order == 4):
        qp = points(5)
        # we = 1/90*np.r_[7,32,12,32,7]
        we = np.array([7/90,32/90,12/90,32/90,7/90])
    if order == 5:
        qp = points(6)
        # we = 1/288*np.r_[19,75,50,50,75,19]
        we = np.array([19/288,75/288,50/288,50/288,75/288,19/288])
    if order == 7:
        qp = points(7)
        # we = 1/840*np.r_[41,216,27,272,27,216,41]
        we = np.array([41/840,216/840,27/840,272/840,27/840,216/840,41/840])
    return qp,we


def dunavant(order):
    if order == 0:
        qp = np.array([[1/3],
                       [1/3]])
        we = np.r_[1]
    if order == 1:
        qp = np.array([[0,1,0],
                       [0,0,1]])
        we = np.r_[1/3,1/3,1/3]
    if order == '1kek':
        qp = np.array([[0,1,0],
                       [0,0,1]])
        
        we = np.r_[1/3,1/3,1/3]
    if order == 2:
        qp = np.array([[2/3,1/6,1/6],
                       [1/6,1/6,2/3]])
        we = np.r_[1/3,1/3,1/3]
    if order == '2l':
        qp = np.array([[0,1,0,1/3],
                       [0,0,1,1/3]])
        we = np.r_[1/12,1/12,1/12,3/4]
    if order == '2m':
        qp = np.array([[1/2,  0,1/2],
                       [  0,1/2,1/2]])
        we = np.r_[1/3,1/3,1/3]
    if order == 3:
        qp = np.array([[1/3,3/5,1/5,1/5],
                       [1/3,1/5,1/5,3/5]])
        we = np.r_[-9/16,25/48,25/48,25/48]
        print('ACTHUNG! negative weights')
    if order == 4:
        qp = np.array([[0.108103018168070,   0.445948490915965,   0.445948490915965,   0.816847572980459,   0.091576213509771,   0.091576213509771], #roots of 4th order polynomials apparently.
                       [0.445948490915965,   0.445948490915965,   0.108103018168070,   0.091576213509771,   0.091576213509771,   0.816847572980459]])
        we = np.r_[0.223381589678011,   0.223381589678011,   0.223381589678011,   0.109951743655322,   0.109951743655322,   0.109951743655322]
    if order == 5:
        qp = np.array([[0.333333333333333,   0.059715871789770,   0.470142064105115,   0.470142064105115,   0.797426985353087,   0.101286507323456,   0.101286507323456],
                       [0.333333333333333,   0.470142064105115,   0.470142064105115,   0.059715871789770,   0.101286507323456,   0.101286507323456,   0.797426985353087]])
        we = np.r_[0.225000000000000,   0.132394152788506,   0.132394152788506,   0.132394152788506,   0.125939180544827,   0.125939180544827,   0.125939180544827]
    if order == 6:
        qp = np.array([[0.501426509658179,   0.249286745170910,   0.249286745170910,   0.873821971016996,   0.063089014491502,   0.063089014491502,   0.053145049844817,   0.310352451033784,   0.636502499121399,   0.310352451033784,   0.636502499121399,   0.053145049844817],
                       [0.249286745170910,   0.249286745170910,   0.501426509658179,   0.063089014491502,   0.063089014491502,   0.873821971016996,   0.310352451033784,   0.636502499121399,   0.053145049844817,   0.053145049844817,   0.310352451033784,   0.636502499121399]])
        we = np.r_[0.116786275726379,   0.116786275726379,   0.116786275726379,   0.050844906370207,   0.050844906370207,   0.050844906370207,   0.082851075618374,   0.082851075618374,   0.082851075618374,   0.082851075618374,   0.082851075618374,   0.082851075618374]
        
    return qp,we

def quadrule(order):
    if order == 0:
        qp = np.array([[1/2],
                       [1/2]])
        we = np.r_[1]
    if order == 1:
        qp = np.array([[0,1,0,1],
                       [0,0,1,1]])
        we = np.r_[1/4,1/4,1/4,1/4]
    if order == '1aq':
        qp = np.array([[1/2,  0,1/2,  1],
                       [  0,1/2,  1,1/2]])
        we = np.r_[1/4,1/4,1/4,1/4]
    if order == 3:
        v1 = 1/6*(3-np.sqrt(3))
        v2 = 1/6*(3+np.sqrt(3))
        qp = np.array([[v1,v2,v1,v2],
                       [v1,v1,v2,v2]])
        we = np.r_[1/4,1/4,1/4,1/4]
    if order == '3l':
        qp = np.array([[0,1,0,1,1/2],
                       [0,0,1,1,1/2]])
        we = np.r_[1/12,1/12,1/12,1/12,2/3]
    if order == 5: # square_rule ([0 0],[1 1],[3 3])
        v1 = 1/10*(5-np.sqrt(15))
        v2 = 1/10*(5+np.sqrt(15))
        qp = np.array([[v1,1/2,v2, v1,1/2, v2,v1,1/2,v2],
                       [v1, v1,v1,1/2,1/2,1/2,v2, v2,v2]])
        we = np.r_[25/324, 10/81, 25/324, 10/81, 16/81, 10/81, 25/324, 10/81, 25/324]

    # if order==6:
    #     qp = np.array([])
    #     we = np.r_[0.030250748321401, 0.056712962962963, 0.056712962962963, 0.030250748321401, 0.056712962962963, 0.106323325752674, 0.106323325752674, 0.056712962962963, 0.056712962962963, 0.106323325752674, 0.106323325752674, 0.056712962962963, 0.030250748321401, 0.056712962962963, 0.056712962962963, 0.030250748321401]
    return qp,we

def keast(order):
    if order == 0:
        qp = np.array([[1/4],
                       [1/4],
                       [1/4]])
        we = np.r_[1]
    
    if order == 1:
        qp = np.array([[0,1,0,0],
                       [0,0,1,0],
                       [0,0,0,1]])
        we = np.r_[1/4,1/4,1/4,1/4]
    
    if order == 2:
        v1 = 1/20*(5+3*np.sqrt(5))
        v2 = 1/20*(5-np.sqrt(5))
        qp = np.array([[v1,v2,v2,v2],
                       [v2,v2,v2,v1],
                       [v2,v2,v1,v2]])
        we = np.r_[1/4,1/4,1/4,1/4]
    
    if order == 3:
        qp = np.array([[0,1,0,0,1/3,1/3,  0,1/3],
                       [0,0,1,0,1/3,  0,1/3,1/3],
                       [0,0,0,1,  0,1/3,1/3,1/3]])
        we = np.r_[1/40,1/40,1/40,1/40,9/40,9/40,9/40,9/40]
    
    if order == 3.5:
        qp = np.array([[1/4,1/2,1/6,1/6,1/6],
                       [1/4,1/6,1/6,1/6,1/2],
                       [1/4,1/6,1/6,1/2,1/6]])
        we = np.r_[-4/5,9/20,9/20,9/20,9/20]
        print('ACTHUNG! negative weights')
    
    if order == 4:
        v1 = 0.568430584196844
        v2 = 0.143856471934385
        w1 = 0.217765069880405
        w2 = 0.021489953413063
        qp = np.array([[v1,v2,v2,v2,  0,1/2,1/2,1/2,  0,  0],
                       [v2,v2,v2,v1,1/2,  0,1/2,  0,1/2,  0],
                       [v2,v2,v1,v2,1/2,1/2,  0,  0,  0,1/2]])
        we = np.r_[w1,w1,w1,w1,w2,w2,w2,w2,w2,w2]
        
    return qp,we

# qp,we = dunavant(order=6)
# print(qp.shape)
# print(we.shape)

# # print(np.sum(we))

# print(qp)