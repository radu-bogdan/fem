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
from pde.tools import condest

# @profile
def pcg(Afuns, f, tol = 1e-5, maxit = 100, pfuns = 1):     
    
    maxit = int(maxit)
    
    if not callable(pfuns):
        pfun = lambda x : pfuns@x
        # splu_pfun = sp.linalg.splu(pfuns,permc_spec='COLAMD')
        # pfun = lambda x : splu_pfun.solve(x)
    else:
        pfun = pfuns
    
    if not callable(Afuns):
        Afun = lambda x : Afuns@x
    else:
        Afun = Afuns
    
    if not isinstance(f,npy.ndarray):
        d = f.A.squeeze()
    else:
        d = f.squeeze()
        
    # print('Cond about',condest(pfuns@Afuns))
        
    w = pfun(d)
    rho = w@d
    err0 = npy.sqrt(rho)
    
    s = w
    u = 0*d.copy()
    
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
        
    
    
    
    