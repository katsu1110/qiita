# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 17:20:26 2018

@author: katsuhisa
"""

# Markov Chain Monte Carlo (MCMC) --simple implementations
# libraries
import numpy as np
import matplotlib.pyplot as plt

# target distribution (multivariate Gaussian)
def target_pdf(type):
    if type == 'round':
        x, y = np.random.multivariate_normal([2,2], [[1,0],[0,1]], 5000).T
    elif type == 'correlated':
        x, y = np.random.multivariate_normal([2,2], [[1, 0.9], [0.9, 1]], 5000).T
    elif type == 'bimodal':
        x, y = 0.5*np.random.multivariate_normal([1,1], [[1,0],[0,1]], 5000).T \
            + 0.5*np.random.multivariate_normal([3,3], [[1,0],[0,1]], 5000).T
    elif type == 'close_bimodal':
        x, y = 0.5*np.random.multivariate_normal([1.25,1.25], [[1,0],[0,1]], 5000).T \
            + 0.5*np.random.multivariate_normal([2.75,2.75], [[1,0],[0,1]], 5000).T
    return x, y

# visualization similar to Fig 1b in the Sanborn & Charter 2016
type = ['round', 'correlated', 'bimodal', 'close_bimodal']
plt.close('all')
fig, axes = plt.subplots(1, 4)
x, y = target_pdf(type[0])
axes[0].plot(x,y,'o')
x, y = target_pdf(type[1])
axes[1].plot(x,y,'o')
x, y = target_pdf(type[2])
axes[2].plot(x,y,'o')
x, y = target_pdf(type[3])
axes[3].plot(x,y,'o')



