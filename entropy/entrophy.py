# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 19:06:08 2017

@author: katsuhisa
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

p = np.linspace(0.001,0.999,num=100)
entropy = -p*np.log(p)

plt.plot(p,entropy)
