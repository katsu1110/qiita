# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:51:50 2018

@author: katsuhisa
"""
# libraries ------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%
# functions -----------------------------
def wright_fisher_simulation(repeat, N, p, q):
    '''simulation of Wright-Fisher model:
        repeat: the number of repeat in simulation
        N: the number of individuals in the population
        p: the probability that allele A is held
        q: the probability that allele B is held'''
    # attribute allele A, B to individuals as 1st generation
    gene = {"A": np.random.choice([0, 1], N, replace=True, p=[1-p, p]),
            "B": np.random.choice([0, 1], N, replace=True, p=[1-q, q])}
    results = gene
    # iterate Wright-Fisher process
    for r in np.arange(repeat):
        p = np.sum(gene["A"]==1)/N
        q = np.sum(gene["B"]==1)/N
        gene = {"A": np.random.choice([0, 1], N, replace=True, p=[1-p, p]),
            "B": np.random.choice([0, 1], N, replace=True, p=[1-q, q])}
        results.append(gene)
    return results

def wright_fisher_animation(results):
    '''animation for the Wright-Fisher model'''
    plt.cla()
    
#%%
# visualize steps
fig = plt.figure()
ani = animation.FuncAnimation(fig, wright_fisher_animation, interval=500)
plt.show()
    