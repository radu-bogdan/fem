#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:33:29 2023

@author: catalinradu
"""

import gmsh
import numpy as npy
from scipy import sparse as sp
# npy.set_printoptions(edgeitems=30, linewidth = 1000000)

def pcg(Afuns,f,tol=1e-5,maxit=100,pfuns=1):
    
    if not callable(pfuns):
        pfun = lambda x : sp.linalg.spsolve(pfuns,x)
    if pfuns == 1:
        pfun = lambda x : sp.linalg.spsolve(sp.identity(f.shape[0]),x)
    
    if not callable(Afuns):
        Afun = lambda x : Afuns@x
    else:
        Afun = Afuns            
    
    maxit = int(maxit)
    
    if not isinstance(f,npy.ndarray):
        d = f.A.squeeze()
    else:
        d = f.squeeze()
        
    w = pfun(d)
    rho = w@d
    err0 = npy.sqrt(rho)
    
    s = w
    u = 0*d
    
    for it in range(maxit):
        As = Afun(s)
        alpha = rho/(As@s)
        u = u + alpha*s
        d = d - alpha*As
        w = pfun(d)
        rho1 = rho
        rho = w@d
        err = npy.sqrt(rho)
        if err < tol*err0:
            break
        beta = rho/rho1
        s = w + beta*s
    print('pcg stopped after ' + str(it) + ' iterations with relres ' + str(err/err0))
    return u#,it,d
        
    
    
    
    