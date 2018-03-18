# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:51:50 2018

Create animation of Wright-Fisher model simulation, which looks similar to
the figure 1c in Suchow et al., 2017, Trends in Cognitive Science

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
    results = [gene]
    # iterate Wright-Fisher process
    for r in np.arange(repeat-1):
        p = np.sum(gene["A"]==1)/N
        q = np.sum(gene["B"]==1)/N
        gene = {"A": np.random.choice([0, 1], N, replace=True, p=[1-p, p]),
            "B": np.random.choice([0, 1], N, replace=True, p=[1-q, q])}
        results.append(gene)
    return results   
    
def allele_color(a, b):
    '''color code for pairs of allele'''
    if a==1 and b==1:
        col = "r"
    elif a==1 and b==0:
        col = "m"
    elif a==0 and b==1:
        col = "c"
    elif a==0 and b==0:
        col = "b"
    return col

def animate(i):
    '''for animation'''
    plt.cla()
    plt.axis('off')
    plt.text(3.5, 6.7, "(P(A)=" + str(pa) + ", ", fontsize=15, color="k", ha="left")
    plt.text(5, 6.7, "P(B)=" + str(pb) + ")", fontsize=15, color="k", ha="left")
    plt.text(-0.5, 6.7, "AB", fontsize=20, color="r", ha="left")
    plt.text(0.5, 6.7, "Ax", fontsize=20, color="m", ha="left")
    plt.text(1.5, 6.7, "xB", fontsize=20, color="c", ha="left")
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
pa = 0.5
pb = 0.5
results = wright_fisher_simulation(ngene, N, pa, pb)
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
        c.append(allele_color(gene["A"][n], gene["B"][n]))
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
