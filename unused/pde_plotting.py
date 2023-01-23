#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 00:27:45 2022

@author: thrall
"""

import numpy
import matplotlib.pyplot as plt
from .init_refine import make_edge_list

def pdemesh(p,t,dpi=500,info=0):
    
    plt.rcParams['figure.dpi'] = 200 # Always do this first ... for some reason
    plt.rcParams['font.size'] = 8
    plt.gca().set_aspect('equal', adjustable='box')
    
    x = numpy.array([p[0,t[0,:]],p[0,t[1,:]],p[0,t[2,:]],p[0,t[0,:]]])
    y = numpy.array([p[1,t[0,:]],p[1,t[1,:]],p[1,t[2,:]],p[1,t[0,:]]])
    
    plt.plot(x,y,'b-',linewidth=0.4)
    
    if info==1:
        for i in range(p.shape[1]):
            plt.text(p[0,i],p[1,i],numpy.r_[0:p.shape[1]][i], \
                      horizontalalignment='center',verticalalignment='center', \
                      backgroundcolor='yellow', \
                      bbox=dict(facecolor='yellow', edgecolor='none', alpha=1, pad=1), \
                      fontsize='x-small')        
        
        avg_x = 1/3*(x[1,:] + x[2,:] + x[3,:])
        avg_y = 1/3*(y[1,:] + y[2,:] + y[3,:])
        for i in range(t.shape[1]):
            plt.text(avg_x[i],avg_y[i],numpy.r_[0:t.shape[1]][i], \
                      horizontalalignment='center',verticalalignment='center', \
                      # backgroundcolor='yellow', \
                      # bbox=dict(facecolor='yellow', edgecolor='none', alpha=1, pad=1), \
                      fontsize='x-small')
    
    if info==1:
        ued,_ = make_edge_list(t)
        edge_mid = 1/2*(p[:,ued[0,:]] + p[:,ued[1,:]])
        for i in range(ued.shape[1]):
            plt.text(edge_mid[0,i],edge_mid[1,i],numpy.r_[0:ued.shape[1]][i], \
                      horizontalalignment='center',verticalalignment='center', \
                      backgroundcolor='gray', \
                      bbox=dict(facecolor='silver', edgecolor='none', alpha=1, pad=0.5), \
                      fontsize='x-small')   
        print('something')


def pdesurf(p,t,dpi=500,info=0):
    return 0


  
