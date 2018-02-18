# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 17:20:26 2018

@author: katsuhisa
"""

# Markov Chain Monte Carlo (MCMC) --simple implementations
# libraries ------------------------------
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# functions -----------------------------
def target_params(disttype):
    """preset parameters for target distributions"""
    if disttype == 'round':
        me = [2,2]
        cov = [[1,0],[0,1]]
    elif disttype == 'correlated':
        me = [2,2]
        cov = [[1,0.9],[0.9,1]]
    elif disttype == 'bimodal':
        me = [[0,0],[6,6]]
        cov = [[1,0],[0,1]]
    elif disttype == 'close_bimodal':
        me = [[0,0],[10,10]]
        cov = [[1,0],[0,1]]
    return me, cov
    
        
def target_rvs(N, disttype):
    """random samples from target distribution (bivariate Gaussian)"""
    me, cov = target_params(disttype)
    if disttype == 'round' or 'correlated':
        p = multivariate_normal(mean=me, cov=cov).rvs(size=N, random_state=1220).T
        x = p[0]
        y = p[1]
    elif disttype == 'bimodal' or 'close_bimodal':
        x = np.array([])
        y = np.array([])
        for w in range(1):
            p = w*multivariate_normal(mean=me[0], cov=cov).rvs(size=round(N/2), random_state=1220).T \
                + (1-w)*multivariate_normal(mean=me[1], cov=cov).rvs(size=int(N-round(N/2)), random_state=1220).T
            x = np.append(x, p[0])
            y = np.append(y, p[1])
    return x, y

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE)"""

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

def target_pdf(p, disttype):
    """target distribution (bivariate Gaussian)"""
    me, cov = target_params(disttype)
    if disttype == 'round' or 'correlated':
        prob = multivariate_normal.pdf(p, mean=me, cov=cov)
    elif disttype == 'bimodal' or 'close_bimodal':
        prob0 = multivariate_normal.pdf(p, mean=me[0], cov=cov)
        prob1 = multivariate_normal.pdf(p, mean=me[1], cov=cov)
        prob = max([prob0, prob1])        
        
    return prob

def proposal_pdf(p):
    """proposal distribution"""
    return (p[0] + np.random.normal(0, 1), p[1] + np.random.normal(0, 1))

def mh(N, disttype):
    """metropolis hastings sampling (no 'thin' and 'burn_in' for simplicity)"""
    xs = np.array([])
    ys = np.array([])
    pos_now = (0,0)
    accept = 0
    for i in range(N):
        pos_cand = proposal_pdf(pos_now)
        prob_stay = target_pdf(pos_now, disttype)
        prob_move = target_pdf(pos_cand, disttype)
        if prob_move / prob_stay > np.random.uniform(0,1,1):
            pos_now = pos_cand
            xs = np.append(xs, pos_now[0])
            ys = np.append(ys, pos_now[1])
            accept += 1
    return xs, ys, accept/N

# visualization similar to the Fig 1b in Sanborn & Charter 2016 -----------
# preset parameters
disttype = ['round', 'correlated', 'close_bimodal', 'bimodal']
plt.close('all')

# probability distributions
fig, axes = plt.subplots(1, 4)
for i in range(4):
    x, y = target_rvs(5000, disttype[i])
    xx, yy, zz = kde2D(x, y, 1)
    axes[i].pcolormesh(xx, yy, zz)
    axes[i].set_title(disttype[i])
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    axes[i].get_xaxis().set_ticks([])
    axes[i].get_yaxis().set_ticks([])
    axes[i].get_xaxis().set_ticklabels([])
    axes[i].get_yaxis().set_ticklabels([])


