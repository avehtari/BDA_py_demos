"""Bayesian Data Analysis, 3rd ed
Chapter 6, demo 4

Posterior predictive checking
Light speed example

"""

from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2)
plt.rc('axes', color_cycle=('#377eb8','#e41a1c','#4daf4a',
                            '#984ea3','#ff7f00','#ffff33'))

# data
data_path = '../utilities_and_data/light.txt'
y = np.loadtxt(data_path)
# sufficient statistics
n = len(y)
s2 = np.var(y, ddof=1)
my = np.mean(y)

# tail area probabilities of marginal predictive distributions
Ty = stats.t.cdf(y, n-1, loc=my, scale=np.sqrt(s2*(1+1/n)))

# ====== plot
plt.hist(Ty, np.arange(0, 1.01, 0.05))
plt.xlim((0,1))
plt.title('Light speed example\ndistribution of marginal posterior p-values')

plt.show()

