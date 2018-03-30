# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:51:50 2018

Create animation of Wright-Fisher model simulation, which looks similar to
the figure 1c in Suchow et al., 2017, Trends in Cognitive Science

Implementation of "Wright-Fisher model"

@author: katsuhisa
"""
# libraries ------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%
# functions -----------------------------
def wright_fisher_simulation(repeat, N, pAA, paa):
    '''simulation of Wright-Fisher model:
        repeat: the number of repeat in simulation
        N: the number of individuals in the population
        pAA: the probability that an individual has AA
        paa: the probability that an individual has aa
        (hence the probability that an individual has Aa
        is 1 - pAA - paa)
     '''   
    # attribute allele to individuals as 1st generation
    results = [[np.random.choice([0, 1, 2], N, replace=True, p=[paa, 1-pAA-paa, pAA])]]
    # iterate Wright-Fisher process
    for r in np.arange(repeat-1)+1:
        pAA = np.sum(results[r-1][0]==2)/N
        pAa = np.sum(results[r-1][0]==1)/N
        paa = np.sum(results[r-1][0]==0)/N
        results.append([np.random.choice([0, 1, 2], N, replace=True, p=[paa, pAa, pAA])])
    return results     
    
def allele_color(x):
    '''color code for pairs of allele'''
    if x==2:
        col = "r"
    elif x==1:
        col = "g"
    elif x==0:
        col = "b"
    return col

def animate(i):
    '''for animation'''
    plt.cla()
    plt.axis('off')
    plt.text(2.3, 6.7, "P(AA), P(Aa), P(aa)=" + str(pAA) + ", " + str(round(100*(1-pAA-paa))/100) +
                         ", " + str(paa), fontsize=12, color="k", ha="left")
    plt.text(-0.5, 6.7, "AA", fontsize=20, color="r", ha="left")
    plt.text(0.3, 6.7, "Aa", fontsize=20, color="g", ha="left")
    plt.text(1.1, 6.7, "aa", fontsize=20, color="b", ha="left")
    plt.text(1.7, -0.9, "generation " + str(i), fontsize=20, color="k", ha="left")
    if i <= ngene:
        im  = plt.scatter(generation[i][0], generation[i][1], marker='o', s=300, 
                          c=generation[i][2], edgecolors='none') 
    else:
        im = plt.scatter(generation[ngene][0], generation[ngene][1], marker='o', s=300, 
                          c=generation[ngene][2], edgecolors='none')     
    return im    
        
#%%
# visualize simulation steps
ngene = 100
N = 49
pAA = 0.45
paa = 0.25
results = wright_fisher_simulation(ngene, N, pAA, paa)
plt.close('all')
fig = plt.figure()
ims = []
ncol = np.floor(np.sqrt(N))
if ncol**2 < N:
    nrow = ncol + 1
else:
    nrow = ncol
generation = []
for g in range(ngene):
    plt.cla()
    x = [0]
    y = [0]
    c = []
    for n in np.arange(N):
        c.append(allele_color(results[g][0][n]))
        if x[-1]+1 >= ncol:
            x.append(0)
            y.append(y[-1]+1)
        else:
            x.append(x[-1]+1)
            y.append(y[-1])
    generation.append((x[:-1], y[:-1], c))

# animation    
ani = animation.FuncAnimation(fig, animate, interval=500)
ani.save('wright_fisher_animation.mp4',writer='ffmpeg')
