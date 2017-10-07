# -*- coding: utf-8 -*-
"""
Created on Sat May 27 08:17:30 2017

@author: katsuhisa
"""

def normalize(v,newmin,newmax):
    a = (newmax - newmin)/(max(v) - min(v))
    b = newmax - a*max(v)

    return [a*i + b for i in v]

# generate a vector with random number
import random
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(1)
v = [random.random() for _ in range(10)]

plt.plot(v, label='raw')
plt.plot(normalize(v,0,1), label='0~1')
plt.plot(normalize(v,-1,1), label='-1~1')
plt.legend()

