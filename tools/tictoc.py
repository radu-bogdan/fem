#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 22:41:56 2022

@author: bogdan
"""

# make class?

import time

timeStamp = 0.0

def tic():
    global timeStamp
    timeStamp = time.process_time()
    return
    
    
def toc():
    global timeStamp
    elapsed = time.process_time()-timeStamp
    timeStamp = 0.0
    return elapsed