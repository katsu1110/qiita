# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""


# import librarys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# get data
iq = np.array([61, 88, 89, 89, 90, 92, 93, 
               94, 98, 98, 101, 102, 105, 108,
               109, 113, 114, 115, 120, 138])

# compute mean, SEM (standard error of the mean) and median
mean_iq = np.average(iq)
sem_iq = np.std(iq)/np.sqrt(len(iq))
median_iq = np.median(iq)

# bootstrap to compute sem of the median
def bootstrap(x,repeats):
    # placeholder (column1: mean, column2: median)
    vec = np.zeros((2,repeats))
    for i in np.arange(repeats):
        # resample data with replacement
        re = np.random.choice(len(x),len(x),replace=True)
        re_x = x[re]
        
        # compute mean and median of the "new" dataset
        vec[0,i] = np.mean(re_x)
        vec[1,i] = np.median(re_x)
    
    # histogram of median from resampled datasets
    sns.distplot(vec[1,:], kde=False)
    
    # compute bootstrapped standard error of the mean,
    # and standard error of the median
    b_mean_sem = np.std(vec[0,:])
    b_median_sem = np.std(vec[1,:])
    
    return b_mean_sem, b_median_sem   

# execute bootstrap
bootstrapped_sem = bootstrap(iq,1000)    



    
## jackknife to compute sem of the median (NO GOOD)
#def jackknife(x):
#    # placeholder (column1: mean, column2: median)
#    vec = np.zeros((2,len(x)))
#    for i in np.arange(len(x)):
#        # resample data by "leave-one-out"
#        x_loo = np.delete(x,i)
#                    
#        # compute mean and median of the "new" dataset
#        vec[0,i] = np.mean(x_loo)
#        vec[1,i] = np.median(x_loo)
#    
#    # histogram of median from resampled datasets
#    sns.distplot(vec[1,:])
#    
#    # compute jackknifed standard error of the mean,
#    # and standard error of the median
#    j_mean_sem = np.std(vec[0,:])
#    j_median_sem = np.std(vec[1,:])
#    
#    return j_mean_sem, j_median_sem
#        
## execute jackknife
#jackknifed_sem = jackknife(iq)
