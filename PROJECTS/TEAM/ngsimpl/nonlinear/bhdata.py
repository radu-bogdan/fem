import numpy as np
from ngsolve import BSpline
import ngsolve as ngs

def BHCurves(nr):
    if abs(nr)==1:
        # nlhfo.fnc from pchstein dipl
        print("BH data from Pechstein nlhfo.fnc")
        H = [0.0, 5.024e2, 6.28e2, 8.792e2, 1.245e3, 2.0096e3, 3.768e3, 6.28e3, 8.792e3, 
            1.256e4, 2.512e4, 3.768e4, 6.28e4, 1.256e5, 1.256e6, 1.256e7, 1.256e8]
        B = [0.0, 1.223992, 1.31579, 1.42986, 1.49938, 1.591168, 1.66254, 1.7135, 1.75426,
            1.8154, 1.958, 2.0598, 2.164, 2.242, 2.442, 2.542, 2.602]
    elif abs(nr)==2: 
        # bh1.dat from pechstein dipl
        print("BH data from Pechstein bh1.dat")
        H = [0.0, 150.0, 300.0, 460.0, 640.0, 720.0, 890.0, 1020.0, 1280.0,
            1900.0, 3000.0, 5000.0, 5.0e4, 2.0e5]
        B = [0.0, 0.21, 0.55, 0.80, 0.95, 1.0, 1.10, 1.15, 1.25,
            1.40, 1.50, 1.60, 1.70, 1.80]
    elif abs(nr)==3: 
        # team problem 20
        # print("BH data from TEAM 20 problem")
        B = [0.0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
            0.80, 0.90, 1.0, 1.1, 1.2, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 
            1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3]
        H = [0.0, 27.0, 58.0, 100.0, 153.0, 185.0, 205.0, 233.0, 255.0, 285.0, 320.0, 355.0, 
            405.0, 470.0, 555.0, 673.0, 836.0, 1065.0, 1220.0, 1420.0, 1720.0, 2130.0, 2670.0, 3480.0, 
            4500.0, 5950.0, 7650.0, 10100.0, 1.3e4, 1.59e4, 2.11e4, 2.63e4, 3.29e4, 4.27e4, 
            6.17e4, 8.43e4, 1.1e5, 1.35e5]
    elif abs(nr)==4:
        # team problem 13 (modified)
        print("BH data from TEAM 13 problem")
        H=[0.000000, 16.000000, 30.000000, 54.000000, 143.000000, 191.000000, 210.000000, 222.000000, 233.000000, 247.000000, 
           258.000000, 272.000000, 289.000000, 313.000000, 342.000000, 377.000000, 433.000000, 509.000000, 648.000000, 933.000000, 
           1228.000000, 1934.000000, 2913.000000, 4993.000000, 7189.000000, 9423.000000, 10434.543233, 13076.024844, 15913.009217, 
           18996.640479, 22405.223203, 26270.093692, 30845.768362, 36780.633915, 48895.609107, 87535.218701, 127323.954474]
        
        B=[0.000000, 0.002500, 0.005000, 0.012500, 0.050000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000, 
           0.800000, 0.900000, 1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000, 1.550000, 1.600000, 1.650000, 1.700000, 
           1.750000, 1.800000, 1.820000, 1.870000, 1.920000, 1.970000, 2.020000, 2.070000, 2.120000, 2.170000, 2.220000, 2.270000, 
           2.320000]

        # H=[0.000000, 16.000000, 30.000000, 54.000000, 143.000000, 191.000000, 210.000000, 222.000000, 233.000000, 247.000000, 
        #    258.000000, 272.000000, 289.000000, 313.000000, 342.000000, 377.000000, 433.000000, 509.000000, 648.000000, 933.000000, 
        #    1228.000000, 1934.000000, 2913.000000, 4993.000000, 7189.000000, 9423.000000, 11786.203168, 26669.513755, 55498.513919, 
        #    99677.989448, 160612.726124, 239707.509732, 338367.126058, 457996.360886, 600000.000000, 1100000.000000, 1600000.000000]
        # B=[0.000000, 0.002500, 0.005000, 0.012500, 0.050000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
        #    0.800000, 0.900000, 1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000, 1.550000, 1.600000, 1.650000, 1.700000, 
        #    1.750000, 1.800000, 1.820000, 1.870000, 1.920000, 1.970000, 2.020000, 2.070000, 2.120000, 2.170000, 2.220000, 2.270000, 
        #    2.320000]

    # elif abs(nr)==5:
    #     mu0 = 1.256636e-6
    #     nu0 = 1/mu0
    #     B = list(np.arange(0,5,0.1))
    #     H = list(nu0*np.arange(0,5,0.1))

    else:
        print("unknown bh curve")
        stop

    Hdata = H.copy()
    Bdata = B.copy()

    # prolongate by linear law 
    L = len(Hdata)-1
    BL = Bdata[L]
    HL = Hdata[L]
    dH = HL-Hdata[L-1]
    dB = BL-Bdata[L-1]
    for i in range(0,1000):
        BL = BL + dB
        HL = HL + dH
        Bdata.append(BL)
        Hdata.append(HL)

    bb = Bdata.copy()
    hh = Hdata.copy()
    
    order = 3
    if nr > 0:
        # print("returning energy")
        bb.insert(0,0)
        return BSpline(order,bb,hh)
    else:
        # print("returning coenergy")
        hh.insert(0,0)
        return BSpline(order,hh,bb)
    

def Brauer():
    
    k1 = 49.4; k2 = 1.46; k3 = 520.6;

    # f= lambda x,y: k1/2/k2*(np.exp(k2*x**2+k2*y**2)-1)+1/2*k3*(x**2+y**2)
    
    # fx= lambda x,y: (k1*np.exp(k2*(x**2+y**2))+k3)*x
    # fy= lambda x,y: (k1*np.exp(k2*(x**2+y**2))+k3)*y
    
    # fxx=lambda x,y: k1*np.exp(k2*(x**2+y**2))+2*x**2*k1*k2*np.exp(k2*(x**2+y**2))+k3
    # fyy=lambda x,y: k1*np.exp(k2*(x**2+y**2))+2*y**2*k1*k2*np.exp(k2*(x**2+y**2))+k3
    # fxy=fyx=lambda x,y: 2*x*y*k1*k2*np.exp(k2*(x**2+y**2))
    # df= lambda x,y:np.array([fx(x,y),fy(x,y)])
    # ddf= lambda x,y:np.array([[fxx(x,y),fxy(x,y)],[fyx(x,y),fyy(x,y)]])

    fun_w = lambda B : k1/2/k2*(ngs.exp(B**2)-1)+1/2*k3*B**2
    fun_dw = lambda B : (k1*ngs.exp(k2*B**2)+k3)*B
    fun_ddw = lambda B : 2*k1*k2*B**2*ngs.exp(k2*B**2) + k1*ngs.exp(k2*B**2) +k3



    return fun_w, fun_dw, fun_ddw

# w1 = BHCurves(-4)
# wd1 = w1.Differentiate()
# wv = []

# import matplotlib.pyplot as plt
# import numpy as np
# # x = np.arange(0,100_000,100)
# x = np.arange(0,1000000,100)

# for i in range(x.size):
#     wv.append(w1(x[i]))

# plt.plot(x,wv,'*')

# ##################################

# w2 = BHCurves(4)
# wd2 = w2.Differentiate()

# wv = []
# wv2 = []

# import matplotlib.pyplot as plt
# import numpy as np
# x = np.arange(-20,20,0.1)

# for i in range(x.size):
#     wv.append(w2(x[i]))
#     wv2.append(wd2(x[i]))

# # plt.plot(x,wv,'*')
# plt.plot(x,wv2,'*')
# print(min(wv2),max(wv2))

# ##################################

# w2 = BHCurves(3)
# wd2 = w2.Differentiate()
# wv = []

# import matplotlib.pyplot as plt
# import numpy as np
# # x = np.arange(0,100_000,100)
# x = np.arange(0,1000000,100)

# for i in range(x.size):
#     wv.append(w2(x[i]))

# plt.plot(x,wv,'*')



# ##################################

# w = BHCurves(-2)
# wv = []

# import matplotlib.pyplot as plt
# import numpy as np
# # x = np.arange(0,100_000,100)
# x = np.arange(0,1000000,100)

# for i in range(x.size):
#     wv.append(w(x[i]))

# # plt.plot(x,wv,'*')



# ##################################

# w = BHCurves(-5)
# wv = []

# import matplotlib.pyplot as plt
# import numpy as np
# x = np.arange(0,1000000,100)

# for i in range(x.size):
#     wv.append(w(x[i]))

# # plt.plot(x,wv,'*')