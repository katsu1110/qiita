# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 14:12:27 2017

@author: katsuhisa
"""

# library
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# A-kun data
p = 42/100
n = 100

# Frequentist -------------------------------
# due to the binomial distribution with the central limit theorem...
mu = 100*p
sigma = np.sqrt(100*p*(1-p))
ci = norm.interval(0.95, loc=mu, scale=sigma)

# Bayesian ----------------------------------
from scipy.stats import beta

# plot beta distribution
fig, ax = plt.subplots(1,1)

def plotBetaPDF(a,b,ax):    
    # range
    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
    
    # visualize
    ax.plot(x, beta.pdf(x, a, b), lw=5, alpha=0.5)
    ax.set_xlabel('theta')
    ax.set_ylabel('P(theta)')

# prior beta distribution
a = 50
b = 50
plotBetaPDF(a,b,ax)
ax.text(0.54,6,'Prior')

# posterior beta distribution
a = 50+42
b = 50+100-42
plotBetaPDF(a,b,ax)
ax.text(0.48,10,'Posterior')

#plt.close('all')

# 95% Bayesian credible interval
bci = beta.interval(0.95,a,b)
print('95% Bayesian credible interval:' + str(bci))
ax.plot(np.array([bci[0],bci[0]]),np.array([0,12]))
ax.plot(np.array([bci[1],bci[1]]),np.array([0,12]))