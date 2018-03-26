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
def wright_fisher_simulation(repeat, N, p):
    '''simulation of Wright-Fisher model:
        repeat: the number of repeat in simulation
        N: the number of individuals in the population
        p: the probability that gene A is held
     '''   
        # attribute allele 1, 1 to individuals as 1st generation
    gene = {"allele1": np.random.choice([0, 1], N, replace=True, p=[1-p, p]),
            "allele2": np.random.choice([0, 1], N, replace=True, p=[1-p, p])}
    results = [gene]
    # iterate Wright-Fisher process
    for r in np.arange(repeat-1):
        p = (np.sum(gene["allele1"]==1) + np.sum(gene["allele2"]==1))/(2*N)
        gene = {"allele1": np.random.choice([0, 1], N, replace=True, p=[1-p, p]),
            "allele2": np.random.choice([0, 1], N, replace=True, p=[1-p, p])}
        results.append(gene)
    return results   
    
def allele_color(a, b):
    '''color code for pairs of allele'''
    if a==1 and b==1:
        col = "r"
    elif (a==1 and b==0) or (a==0 and b==1):
        col = "g"
    elif a==0 and b==0:
        col = "b"
    return col

def animate(i):
    '''for animation'''
    plt.cla()
    plt.axis('off')
    plt.text(3.5, 6.7, "(P(A)=" + str(pa) + ")", fontsize=12, color="k", ha="left")
    plt.text(0.5, 6.7, "AA", fontsize=20, color="r", ha="left")
    plt.text(1.5, 6.7, "Ax", fontsize=20, color="g", ha="left")
    plt.text(2.5, 6.7, "xx", fontsize=20, color="b", ha="left")
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
pa = 0.6
results = wright_fisher_simulation(ngene, N, pa)
plt.close('all')
fig = plt.figure()
ims = []
ncol = np.floor(np.sqrt(N))
if ncol**2 < N:
    nrow = ncol + 1
else:
    nrow = ncol
generation = []
for gene in results:
    plt.cla()
    x = [0]
    y = [0]
    c = []
    for n in np.arange(N):
        c.append(allele_color(gene["allele1"][n], gene["allele2"][n]))
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
