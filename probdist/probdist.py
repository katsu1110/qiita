# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 09:29:38 2017

@author: katsuhisa
"""


## Bernoulli distribution ---------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli

# probability that X=1
p = 0.3

# possible outcomes
x = [0,1]

# visualize
fig, ax = plt.subplots(1,1)
ax.plot(x, bernoulli.pmf(x, p), 'bo', ms=8)
ax.vlines(x, 0, bernoulli.pmf(x,p), colors='b', lw=5, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('probability')
ax.set_title('bernoulli pmf')
ax.set_ylim((0,1))
ax.set_aspect('equal')

plt.close('all')

## Binomial distribution ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom

# probability that X=1
p = 0.3

# the number of trials
N = 5
 
# X=1 happens k times
k = np.arange(N+1)

# visualize
fig, ax = plt.subplots(1,1)
ax.plot(k, binom.pmf(k, N, p), 'bo', ms=8)
ax.vlines(k, 0, binom.pmf(k, N, p), colors='b', lw=5, alpha=0.5)
ax.set_xlabel('k')
ax.set_ylabel('probability (X=1)')
ax.set_title('binomial pmf')
ax.set_ylim((0,1))

plt.close('all')

## Poisson distribution --------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# rate parameter
mu = 5

# possible counts
k = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))

# visualize
fig, ax = plt.subplots(1,1)
ax.plot(k, poisson.pmf(k, mu), 'bo', ms=8)
ax.vlines(k, 0, poisson.pmf(k, mu), colors='b', lw=5, alpha=0.5)
ax.set_xlabel('counts')
ax.set_ylabel('probability')
ax.set_title('poisson pmf')

plt.close('all')


## Exponential distribution --------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon

# rate parameter
mu = 5

# possible waiting time
k = np.linspace(expon.ppf(0.01, loc=0, scale=1/mu), expon.ppf(0.99, loc=0, scale=1/mu), 100)

# visualize
fig, ax = plt.subplots(1,1)
ax.plot(k, expon.pdf(k, loc=0, scale=1/mu), 'r-', lw=5, alpha=0.5)
ax.set_xlabel('waiting time')
ax.set_ylabel('f(x)')
ax.set_title('exponential pdf')

plt.close('all')



## Uniform distribution ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform

# range
a = 3
b = 4 - a
x = np.linspace(uniform.ppf(0.01, loc=a, scale=b), uniform.ppf(0.99, loc=a, scale=b), 100)

# visualize
fig, ax = plt.subplots(1,1)
ax.plot(x, uniform.pdf(x, loc=a, scale=b), 'r-', alpha=0.6, lw=5)
ax.set_xlabel('X')
ax.set_ylabel('f(X)')
ax.set_title('uniform pdf')


## Gamma distribution --------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma

# total number of events
n = 7

# rate parameter
mu = 5

# possible waiting time
x = np.linspace(gamma.ppf(0.01, n, loc=0, scale=1/mu), gamma.ppf(0.99, n, loc=0, scale=1/mu), 100)

# visualize
fig, ax = plt.subplots(1,1)
ax.plot(x, gamma.pdf(x, n, loc=0, scale=1/mu), 'r-', lw=5, alpha=0.5)
ax.set_xlabel('total waiting time until the neuron fires 7 times')
ax.set_ylabel('f(x)')
ax.set_title('gamma pdf')

plt.close('all')


## Beta distribution --------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta

# parameters
a = 0.5
b = 0.5

# range
x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)

# visualize
fig, ax = plt.subplots(1,1)
ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('f(x)')
ax.set_title('beta pdf')

plt.close('all')


## Normal distribution --------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# mean
mu = 0

# standard deviation
sd = 1

# range
x = np.linspace(norm.ppf(0.01, loc=mu, scale=sd), norm.ppf(0.99, loc=mu, scale=sd), 100)

# visualize
fig, ax = plt.subplots(1,1)
ax.plot(x, norm.pdf(x, loc=mu, scale=sd), 'r-', lw=5, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('f(x)')
ax.set_title('normal pdf')

plt.close('all')


## t distribution --------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t

# degree of freedom
df = 15

# range
x = np.linspace(t.ppf(0.01, df), t.ppf(0.99, df), 100)

# visualize
fig, ax = plt.subplots(1,1)
ax.plot(x, t.pdf(x, df), 'r-', lw=5, alpha=0.5)
ax.set_xlabel('y')
ax.set_ylabel('f(y)')
ax.set_title('t pdf')

plt.close('all')