import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../mixed.EM') # adds parent directory

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