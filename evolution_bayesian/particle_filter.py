# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:47:13 2018

Create a figure, which looks similar to the figure 2
in Suchow et al., 2017, Trends in Cognitive Science

Implementation of "Particle filter" along with the following:
https://qiita.com/kenmatsu4/items/c5232b1499dfd00e877d

@author: katsuhisa
"""

# libraries ------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# artificial data for a working memory task (n_object = 6)
lam = 8
t = np.arange(1,15)
n_remembered = lam/t + 0.05*np.random.rand(len(t))

#%%
# functions -----------------------------
class ParticleFilter(object):
    def __init__(self, y, n_particle, sigma_2):
        self.y = y
        self.n_particle = n_particle
        self.sigma_2 = sigma_2
        self.log_likelihood = -np.inf

    def norm_likelihood(self, y, x, s2):
        return (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-(y-x)**2/(2*s2))

    def F_inv(self, w_cumsum, idx, u):
        if np.any(w_cumsum < u) == False:
            return 0
        k = np.max(idx[w_cumsum < u])
        return k+1

    def resampling(self, weights):
        idx = np.asanyarray(range(self.n_particle))
        u0 = np.random.uniform(0, 1/self.n_particle)
        u = [1/self.n_particle*i + u0 for i in range(self.n_particle)]
        w_cumsum = np.cumsum(weights)
        k = np.asanyarray([self.F_inv(w_cumsum, idx, val) for val in u])
        return k

    def simulate(self, seed=1220):
        np.random.seed(seed)

        # the number of data points in the time-series
        T = len(self.y)

        # x as a hidden state variable
        x = np.zeros((T+1, self.n_particle))
        x_resampled = np.zeros((T+1, self.n_particle))

        # initialization of x as the number of items to be remembered
#        initial_x = np.random.normal(0, 1, size=self.n_particle)
        initial_x = self.y[0] + np.random.normal(0, self.sigma_2, size=self.n_particle)
        x_resampled[0] = initial_x
        x[0] = initial_x

        # weight
        w        = np.zeros((T, self.n_particle))
        w_normed = np.zeros((T, self.n_particle))

        l = np.zeros(T) # initialization of likelihood in each time point
        for t in range(T):
            print("\r calculating... t={}".format(t), end="")
            for i in range(self.n_particle):
                # v as a system noise
                v = np.random.normal(0, self.sigma_2) 
                x[t+1, i] = x_resampled[t, i] + v # add the system noise
                # weights as particles' individual likelihood
                w[t, i] = self.norm_likelihood(self.y[t], x[t+1, i], self.sigma_2) 
            w_normed[t] = w[t]/np.sum(w[t]) # normalization
            l[t] = np.log(np.sum(w[t])) # log likelihood

            # indices to be resampled
            k = self.resampling(w_normed[t]) 
            x_resampled[t+1] = x[t+1, k]

        # overall log likelihood
        self.log_likelihood = np.sum(l) - T*np.log(self.n_particle)
        self.x = x
        self.x_resampled = x_resampled
        self.w = w
        self.w_normed = w_normed
        self.l = l

    def get_filtered_value(self):
        """
        filtered values based on weights
        """
        return np.diag(np.dot(self.w_normed, self.x[1:].T))

    def draw_graph(self):
        T = len(self.y)
        plt.figure(figsize=(6,6))
        plt.plot(range(T), self.y, "ob", markersize=8, alpha=0.4)
        plt.plot(self.get_filtered_value(), "g", linewidth=6, alpha=0.4)
        plt.legend(['data', 'fit'])
        plt.scatter(np.ones(self.n_particle)*0, self.x[0], color="r", 
                        s=6, alpha=0.1)
        for t in range(len(self.w_normed)):
            # reflect the relative likelihood to the particle's marker size
            zscore = (self.w_normed[t] - np.mean(self.w_normed[t]))/np.std(self.w_normed[t])
            for e in range(self.n_particle):
                plt.scatter(t, self.x[t+1][e], color="r", 
                    s=6*(1 + zscore[e]), alpha=0.1)
        
        plt.plot(range(T), np.zeros(T), '--k', alpha=0.1)
        
#%%
# preset hyperparameters
n_particle = 48
sigma_2 = 2

# apply a particle filter and visualize its result
pf = ParticleFilter(n_remembered, n_particle, sigma_2)
pf.simulate()
plt.close('all')
pf.draw_graph()
plt.xlabel('time')
plt.ylabel('the number of remembered items')
plt.ylim(-4.8,14.8)
plt.yticks([0,5,10])
plt.show()