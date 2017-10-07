# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 19:06:08 2017

@author: katsuhisa
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Entropy as a function of probability --------
# random sample for probabilities
p = np.arange(0.01,1,0.01)

# compute binary entropy
entropy = -p*np.log2(p) - (1-p)*np.log2(1-p)

# visualize
plt.plot(p,entropy,linewidth=4,color='r',alpha=0.5)
plt.ylabel('Entropy H(p)', fontsize=18)
plt.xlabel('Probability p', fontsize=18)
plt.tick_params(labelsize=18)
plt.tight_layout()
