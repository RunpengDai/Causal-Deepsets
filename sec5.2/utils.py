# Packages
import scipy as sp
import torch
import torch.nn as nn
# Random
from numpy.random import seed as npseed
from numpy import absolute as np_abs
from numpy.random import normal as rnorm
from numpy.random import uniform as runi
from numpy.random import binomial as rbin
from numpy.random import poisson as rpoisson
from numpy.random import shuffle,randn, permutation # randn(d1,d2) is d1*d2 i.i.d N(0,1)
from numpy.random import lognormal as rlogN
from numpy import squeeze
from numpy.linalg import solve
from matplotlib import pyplot as plt
import pickle

# Numpy
import numpy as np
from numpy import mean, var, std, median
from numpy import array as arr
from numpy import sqrt, cos, sin, exp, dot, diag, ones, identity, quantile, zeros, roll, multiply, stack, concatenate
from numpy import concatenate as v_add
from numpy.linalg import norm, inv
from numpy import apply_along_axis as apply


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from scipy.spatial.distance import pdist as pdist
from scipy.stats import binom
from matplotlib.transforms import Bbox as Bbox
np.set_printoptions(precision = 4)
simple = False

def getAdjGrid(l):
    """
    simple: only 4 neigh
    
    """
    N = l ** 2
    adj_mat = zeros((N, N))
    for i in range(N):
        row = i // l
        col = i % l
        adj_mat[i][i] = 1
        if row != 0:
            adj_mat[i][i - l] = 1
            if not simple:
                if col != 0:
                    adj_mat[i][i - l - 1] = 1
                if col != l - 1:
                    adj_mat[i][i - l + 1] = 1
        if row != l - 1:
            adj_mat[i][i + l] = 1
            if not simple:
                if col != 0:
                    adj_mat[i][i + l - 1] = 1
                if col != l - 1:
                    adj_mat[i][i + l + 1] = 1
        if col != 0:
            adj_mat[i][i - 1] = 1
        if col != l - 1:
            adj_mat[i][i + 1] = 1
    return adj_mat


def to_tensor(x, device):
    if type(x) == list:
        return [torch.from_numpy(i.astype('float32')).to(device) for i in x]
    return torch.from_numpy(x.astype('float32')).to(device)

def to_numpy(x):
    return x.detach().cpu().numpy()

def mse(esti,real):
    return np.mean(np.square(np.array(esti)-np.array(real)))

def ci(esti,real):
    return np.std(np.square(np.array(esti)-np.array(real)))/np.sqrt(len(esti)) 

def draw_results():
    fig = plt.figure(figsize=(20, 5))
    for num,l in enumerate(["l5", "l10", "l15"]):
        dirs = ["results/"+dir for dir in os.listdir("results/") if l in dir]
        if len(dirs) == 0:
            continue
        ax = fig.add_subplot(1,3,num+1)
        false,true = [],[]
        for dir in dirs:
            with open(dir,'rb') as f:
                b = pickle.load(f)
            false.append(b) if "False" in dir else true.append(b)
        false,true = np.array(false).reshape(len(false), 6), np.array(true).reshape(len(true), 6)
        false = false[false[:,0].argsort()]
        true = true[true[:,0].argsort()]

        xt, xf = true[:,0], false[:,0]
        ax.plot(xf,false[:,1],color='darkred', markerfacecolor='none', marker='o', markersize =5,label = "Mean DR")
        ax.fill_between(xf, false[:,1] - false[:,3], false[:,1] + false[:,3], color='darkred', alpha=0.2)
        
        ax.plot(xf,false[:,2],color='darkgreen', markerfacecolor='none', marker='o', markersize =5,label = "Mean PLG")
        ax.fill_between(xf, false[:,2] - false[:,4], false[:,2] + false[:,4], color='darkgreen', alpha=0.2)

        ax.plot(xt,true[:,1],color='magenta', markerfacecolor='none', marker='o', markersize =5,label = "Deep DR")
        ax.fill_between(xt, true[:,1] - true[:,3], true[:,1] + true[:,3], color='magenta', alpha=0.2)
        
        ax.plot(xt,true[:,2],color='limegreen', markerfacecolor='none', marker='o', markersize =5,label = "Deep PLG")
        ax.fill_between(xt, true[:,2] - true[:,4], true[:,2] + true[:,4], color='limegreen', alpha=0.2)
    fig.legend()
    fig.savefig("results.png")