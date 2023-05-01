# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:25:50 2023

@author: Michaelds
"""
import numpy as np
import scipy.sparse as sps
# from sksparse.cholmod import cholesky as chol
import pde

def GradientDescent(J,dJ,x0,maxIter=1000,eps=0.0000001, printoption=1):  
    x=x0
    flag=0
    for i in range(maxIter):
        sx=-dJ(x)
        F = lambda alpha:J(x+alpha*sx)
        dF = lambda alpha: np.dot(dJ(x+alpha*sx),sx)
        #alpha=WolfePowell(F,dF)
        alpha=AmijoBacktracking(F,dF)
        x=x+alpha*sx
        if printoption:
            print ("GRADIENT DESCEND: Iteration: %2d" %(i+1)+"||obj: %2e" % (J(x))+"|| ||grad||: %2e" % (np.linalg.norm(dJ(x)))+"||alpha: %2e" % (alpha))
        if(np.linalg.norm(dJ(x))<eps):
            flag=1
            return x,flag
    return x,flag

def Newton(J,dJ,ddJ,x0,maxIter=1000,eps=0.0000001, printoption=1):#performs maxIterSteps of Newton or until |df|<eps
    epsangle=0.00001;
    x=x0
    flag=0
    angleCondition=np.zeros(5);
    for i in range(maxIter):
        sx=np.linalg.solve(ddJ(x),-dJ(x))
        if (np.dot(sx,-dJ(x))/np.linalg.norm(sx)/np.linalg.norm(dJ(x))<epsangle):
            angleCondition[i%5]=1      
            if np.product(angleCondition)>0:
                sx=-dJ(x)
                print("STEP IN NEGATIVE GRADIENT DIRECTION")
        else:
            angleCondition[i%5]=0
        F  = lambda alpha:J(x+alpha*sx)
        dF = lambda alpha: np.dot(dJ(x+alpha*sx),sx)
        alpha=AmijoBacktracking(F,dF)
        x=x+alpha*sx
        if printoption:
            print ("NEWTON: Iteration: %2d" %(i+1)+"||obj: %2e" % (J(x))+"|| ||grad||: %2e" % (np.linalg.norm(dJ(x)))+"||alpha: %2e" % (alpha))
        if(np.linalg.norm(dF(alpha))<eps):
            flag=1
            return x,flag
    return x,flag

def AmijoBacktracking(F,dF,factor=1/2,mu=0.00001):
    alpha=1
    dphi=dF(0.)
    phi=F(0.)
    for i in range(1000):
        if F(alpha)<=phi+dphi*mu*alpha+phi*np.finfo(float).eps:
            return alpha
        else:
            alpha=alpha*factor
    return alpha;

# NewtonSparse(f, g, h, x0 = u, use_chol = 2, maxIter = 100, printoption = 1)[0]
def NewtonSparse(J,dJ,ddJ,x0,maxIter=100,eps=1e-8,printoption=1,use_chol=0):#performs maxIterSteps of Newton or until |df|<eps
    
    if use_chol == 0:
        solve_ddJ = lambda x,y : sps.linalg.spsolve(ddJ(x),y)
    if use_chol == 1:
        solve_ddJ = lambda x,y : chol(ddJ(x)).solve_A(y)
    if use_chol == 2:
        # solve_ddJ = lambda x,y : pde.pcg(ddJ(x), y, tol = 1e-5, maxit = 10000, pfuns = lambda z : sps.diags(1/(ddJ(x)).diagonal())@z)
        # solve_ddJ = lambda x,y : pde.pcg(ddJ(x), y, tol = 1e-5, maxit = 10000, pfuns = sps.diags(1/(ddJ(x)).diagonal()))
        solve_ddJ = lambda x,y : pde.pcg(ddJ(x), y, tol = 1e-5, maxit = 10000, pfuns = sps.diags(1/(10**7*ddJ(x)).diagonal()))
        # solve_ddJ = lambda x,y : pde.pcg(ddJ(x), y, tol = 1e-5, maxit = 10000, pfuns = sps.tril(ddJ(x), format='csc'))
        
    epsangle = 1e-5;
    x = x0
    flag = 0
    angleCondition = np.zeros(5)
    
    for i in range(int(maxIter)):
        dJx=dJ(x)
        sx=solve_ddJ(x,-dJx)
        # sx = sps.linalg.spsolve(ddJ(´x),-dJx)
        if (np.dot(sx,-dJx)/np.linalg.norm(sx)/np.linalg.norm(dJx)<epsangle):
            angleCondition[i%5]=1      
            if np.product(angleCondition)>0:
                sx=-dJx
                print("STEP IN NEGATIVE GRADIENT DIRECTION")
        else:
            angleCondition[i%5]=0
        #F = lambda alpha:J(x+alpha*sx)
        #dF = lambda alpha: np.dot(dJ(x+alpha*sx),sx)
        dJsx= lambda alpha: dJ(x+alpha*sx)
        #alpha=WolfePowell(F,dF)
        # alpha=AmijoBacktracking(F, dF)
        alpha=ResidualLinesearch(dJsx)
        x=x+alpha*sx
        if printoption:
            print ("NEWTON: Iteration: %2d" %(i+1)+"||obj: %2e" % (J(x))+"|| ||grad||: %2e" % (np.linalg.norm(dJ(x)))+"||alpha: %2e" % (alpha))
        if(np.linalg.norm(np.linalg.norm(dJ(x)))<eps):
            flag=1
            return x,flag
    return x,flag

def ResidualLinesearch(dJsx,factor=1/2,mu=0.01):
    alpha=1
    for i in range(1000):
        if np.linalg.norm(dJsx(alpha))<=np.linalg.norm(dJsx(0)):
            return alpha
        else:
            alpha=alpha*factor
    return alpha;

def WolfePowell(F,dF): # Finds proper line search parameter 
    mu = 0.01; sigma = 0.9; tau = 0.1; tau1 = 0.1; tau2 = 0.6; zeta1 = 1; zeta2 = 10; alpha = 1;
    phi_min = 0
    # 0 < mu < 1/2
    # mu < sigma < 1
    # 0 < tau < 1/2
    # 0 < tau1 < tau2 < 1
    # 1 \le zeta_1 \le zeta_2
    # alpha0>0
    alphaL = 0
    phiL = F(0)
    dphiL = dF(0)
    flag = 1
    
    for i in range(100):
        phi_hat = F(alpha)
        if phi_hat < phi_min:
            return alpha
        if phi_hat > F(0) + mu*alpha*dF(0):
            flag = 0
            alphaR = alpha
            delta = alphaR-alphaL
            c = (phi_hat - phiL -dF(alphaL)*delta)*1/(delta**2)
            alpha_welle = alphaL-dF(alphaL)/(2*c)
            alpha = min(max(alphaL + tau*delta,alpha_welle),alphaR-tau*delta)
        else:
            dphi_hat = dF(alpha)
            if dphi_hat < sigma*dF(0):
                if flag == 1:
                    if dphiL/dphi_hat > (1 + zeta2)/zeta2:
                        alpha_welle = alpha + (alpha-alphaL)*max(dphi_hat/(dphiL-dphi_hat),zeta1)
                    else:
                        alpha_welle = alpha + zeta2*(alpha-alphaL)
                else:
                    if dphiL/dphi_hat > 1 + (alpha-alphaL)/(tau2*(alphaR-alpha)):
                        alpha_welle = alpha + max((alpha-alphaL)*dphi_hat/(dphiL-dphi_hat),tau1*(alphaR-alpha))
                    else:
                        alpha_welle = alpha + tau2*(alphaR-alpha)
                        
                alphaL = alpha; phiL = phi_hat; dphiL = dphi_hat; alpha = alpha_welle;
            else:
                return alpha