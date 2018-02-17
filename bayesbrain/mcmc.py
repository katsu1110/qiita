# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 17:20:26 2018

@author: katsuhisa
"""

# Markov Chain Monte Carlo (MCMC) --simple implementations
# libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def target_pdf(type):
    """target distribution (multivariate Gaussian)"""
    if type == 'round':
        x, y = np.random.multivariate_normal([2,2], [[1,0],[0,1]], 5000).T
    elif type == 'correlated':
        x, y = np.random.multivariate_normal([2,2], [[1, 0.9], [0.9, 1]], 5000).T
    elif type == 'close_bimodal':
        b = [0, 1]
        x = np.array([])
        y = np.array([])
        for w in b:
            xb, yb = w*np.random.multivariate_normal([0,0], [[1,0],[0,1]], 5000).T \
                + (1-w)*np.random.multivariate_normal([7.5, 7.5], [[1,0],[0,1]], 5000).T
            x = np.append(x, xb)
            y = np.append(y, yb)
    elif type == 'bimodal':
        b = [0, 1]
        x = np.array([])
        y = np.array([])
        for w in b:
            xb, yb = w*np.random.multivariate_normal([0,0], [[1,0],[0,1]], 5000).T \
                + (1-w)*np.random.multivariate_normal([10, 10], [[1,0],[0,1]], 5000).T
            x = np.append(x, xb)
            y = np.append(y, yb)
    return x, y

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)

# visualization similar to Fig 1b in the Sanborn & Charter 2016
type = ['round', 'correlated', 'close_bimodal', 'bimodal']
plt.close('all')
fig, axes = plt.subplots(1, 4)
for i in range(4):
    x, y = target_pdf(type[i])
    xx, yy, zz = kde2D(x, y, 1)
    axes[i].pcolormesh(xx, yy, zz)
    axes[i].set_title(type[i])
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    axes[i].get_xaxis().set_ticks([])
    axes[i].get_yaxis().set_ticks([])
    axes[i].get_xaxis().set_ticklabels([])
    axes[i].get_yaxis().set_ticklabels([])


