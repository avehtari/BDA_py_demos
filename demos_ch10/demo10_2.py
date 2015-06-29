"""Bayesian data analysis
Chapter 10, demo 2

Importance sampling example

"""

from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2, markeredgewidth=0)
plt.rc('axes', color_cycle=('#377eb8','#e41a1c','#4daf4a',
                            '#984ea3','#ff7f00','#ffff33'))
plt.rc('patch', facecolor='#bfe2ff')

# fake interesting distribution
x = np.linspace(-3, 3, 200)
r = np.array([ 1.1 ,  1.3 , -0.1 , -0.7 ,  0.2 , -0.4 ,  0.06, -1.7 ,
               1.7 ,  0.3 ,  0.7 ,  1.6 , -2.06, -0.74,  0.2 ,  0.5 ])
# Estimate the density (named q, to emphesize that it does not need to be
# normalized). Parameter bw_method=0.48 is used to mimic the outcome of the
# kernelp function in Matlab.
q_func = stats.gaussian_kde(r, bw_method=0.48)
q = q_func.evaluate(x)

# importance sampling example
g = stats.norm.pdf(x)
w = q/g
r = np.random.randn(100)
r = r[np.abs(r) < 3] # remove samples out of the grid
wr = q_func.evaluate(r)/stats.norm.pdf(r)

# plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10,8))
axes[0].plot(x, q, label=r'$q(\theta|y)$')
axes[0].plot(x, g, label=r'$g(\theta)$')
axes[0].set_yticks(())
axes[0].set_title('target and proposal distributions')
axes[0].legend()
axes[1].plot(x, w, label=r'$q(\theta|y)/g(\theta)$')
axes[1].set_title('samples and importance weights')
axes[1].vlines(r, 0, wr, color='#377eb8', alpha=0.4)
axes[1].set_ylim((0,axes[1].get_ylim()[1]))
axes[1].legend()

plt.show()
