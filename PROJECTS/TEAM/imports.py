try:
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x   # if it's not defined simply ignore the decorator.

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numba as nb
import ngsolve as ng
import netgen.occ as occ
import time

def println(x):
    print(x, end =" ")
    
import sys
# sys.path.insert(0,'../../../') # adds parent directory
sys.path.insert(0,'../../') # adds parent directory

import numpy as np
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
from sksparse.cholmod import cholesky as chol

import plotly.io as pio
pio.renderers.default = 'browser'
import dill
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import ngsolve as ng
cmap = plt.cm.jet

from scipy.sparse import bmat