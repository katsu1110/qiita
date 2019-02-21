# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:10:22 2019

simulation on the recent cognitive aging theory

- being old = more memory
 => slower computation
 => more accurate estimate of the true state
 
@author: katsuhisa
"""

# import libraries
import numpy as np
from scipy.stats import expon
import time
import matplotlib.pyplot as plt
import seaborn as sns

# sample memorized objects
def obtain_memory(N, m, n_ratio):
    '''    N : the number of objects in the system
           m : rate parameter for the exponential function
           n_ratio : the ratio of memorized objects'''
    return expon.rvs(loc=0, scale=1/m, size=np.round(N*n_ratio), random_state=None)

# test 
def test(N, m, item_list, n_test):
    ''' N : the number of objects in the system
        item_list : memorized items
        delta : threshold for the acceptance 
        n_test : the number of repetition '''
    test_obj = expon.rvs(loc=0, scale=1/m, size=n_test, random_state=None)
    rt = np.zeros(n_test)
    ans = np.zeros(n_test)
    c = 0
    for n in test_obj:
        tic = time.clock() 
        ans[c] = np.amin(np.absolute(item_list - n))
        toc = time.clock() 
        rt[c] = toc - tic
        
    return np.log10(np.mean(rt)), np.log10(np.mean(ans))

# Monte-Carlo simulation
repeat = 1000
old = np.zeros((repeat, 2))
young = np.zeros((repeat, 2))
N = 10000
m = 5
n_ratio_old = 40
n_ratio_young = 20
n_test = 10
for r in np.arange(repeat):
    # initialize
    old_list = obtain_memory(N, m, n_ratio_old)
    young_list = obtain_memory(N, m, n_ratio_young)

    # test
    old[r, :] = test(N, m, old_list, n_test)
    young[r, :] = test(N, m, young_list, n_test)
    
# visualization
fig, ax = plt.subplots(1,2)
ylabs = ['Processing duration', 'Estimation error']
for k in np.arange(2):
    me = np.array([np.mean(young[:, k], axis=0), np.mean(old[:, k], axis=0)])
    SEM = np.array([np.std(young[:, k], axis=0), np.std(old[:, k], axis=0)])
    ax[k].bar(np.arange(2), me, yerr=SEM, align='center', color='green', ecolor='black')
    ax[k].set_xticks(np.arange(2))
    ax[k].set_xticklabels(('young', 'old'))
    ax[k].set_ylabel(ylabs[k])
    
fig.tight_layout()
plt.show()    

    

