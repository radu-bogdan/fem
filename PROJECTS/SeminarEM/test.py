##########################################################################################
@nb.njit((float64[:],float64[:]),fastmath=True, cache=True)
def gx_gy_nonlinear(x,y):
    
    le = x.size
    
    Bn = np.zeros((2,le),float64)
    
    for k in nb.prange(le):
        Hnk0 = x[k]; Hnk1 = y[k]; 
        # Bnk = np.array([1,1],float64)
        # Hnk = np.array([x[k],y[k]],float64)
        # Bnk = np.linalg.norm(Bn[:,0])*Hnk/np.linalg.norm(Hnk)
        
        nHnk = np.sqrt(Hnk0**2+Hnk1**2)
        Bnk0 = 0*Hnk0/nHnk
        Bnk1 = 0*Hnk1/nHnk
        
        for it in range(10000):
            fxx = fxx_nonlinear(Bnk0,Bnk1); fxy = fxy_nonlinear(Bnk0,Bnk1)
            fyx = fyx_nonlinear(Bnk0,Bnk1); fyy = fyy_nonlinear(Bnk0,Bnk1)
            fx  = fx_nonlinear(Bnk0,Bnk1);  fy  = fy_nonlinear(Bnk0,Bnk1)            
            
            alpha = 1
            
            inv_detF = 1/(fxx*fyy-fxy*fyx)
            inv_F_xx =-inv_detF*fyy; inv_F_xy = inv_detF*fxy
            inv_F_yx = inv_detF*fyx; inv_F_yy =-inv_detF*fxx
            
            w0 = inv_F_xx*(Hnk0-fx)+inv_F_xy*(Hnk1-fy)
            w1 = inv_F_yx*(Hnk0-fx)+inv_F_yy*(Hnk1-fy)
        
            # w = inv_Fxx@(Hnk-Fx)
            
            for r in range(1000):
                fxu = fx_nonlinear(Bnk0-alpha*w0,Bnk1-alpha*w1)
                fyu = fy_nonlinear(Bnk0-alpha*w0,Bnk1-alpha*w1)
                
                if (fxu-Hnk0)**2+(fyu-Hnk1)**2<=(fx-Hnk0)**2+(fy-Hnk1)**2: break                    
                else: alpha = alpha*(1/2)
                    
            
            Bnku0 = Bnk0 - alpha*w0
            Bnku1 = Bnk1 - alpha*w1
            
            if np.abs(Bnku0-Bnk0)+np.abs(Bnku1-Bnk1)<1e-10:
                print(it)
                break
            
            Bnk0 = Bnku0
            Bnk1 = Bnku1
            
            if it==(10000-1):
                print("did not converge!")
        Bn[0,k] = Bnk0
        Bn[1,k] = Bnk1
    return Bn

##########################################################################################