"""Becs-114.1311 Introduction to Bayesian Statistics
Chapter 5, demo 2

Hierarchical model for SAT-example data (BDA3, p. 102)

"""

from __future__ import division
import numpy as np
from scipy.stats import norm
import scipy.io # For importing a matlab file
import matplotlib.pyplot as plt

# edit default plot settings
plt.rc('font', size=14)
plt.rc('lines', color=(0.3,0.5,0.8), linewidth=2)
plt.rc('axes', color_cycle=(plt.rcParams['lines.color'],)) # Disable color cycle

# SAT-example data (BDA3 p. 120)
# y is estimated treatment effect
# s is standard error of effect estimate
y = np.array([28,  8, -3,  7, -1,  1, 18, 12])
s = np.array([15, 10, 16, 11,  9, 11, 10, 18])
M = len(y)


# plot separate and pooled model
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8,10))
x = np.linspace(-40, 60, 500)

# separate
lines = axes[0].plot(x, norm.pdf(x[:,None], y[1:], s[1:]), linewidth=1)
line, = axes[0].plot(x, norm.pdf(x, y[0], s[0]), 'r')
axes[0].legend((line, lines[1]), ('school A', 'other schools'))
axes[0].set_yticks(())
axes[0].set_title('separate model')

# pooled
axes[1].plot(
    x,
    norm.pdf(
        x,
        np.sum(y/s**2)/np.sum(1/s**2),
        np.sqrt(1/np.sum(1/s**2))
    ),
    'r',
    label='All schools'
)
axes[1].legend()
axes[1].set_yticks(())
axes[1].set_title('pooled model')

# hierarchical
# load the pre-computed results for the hierarchical model
# replace this with your own code in Ex 5.1*
hier_res = scipy.io.loadmat('demo5_2.mat')
''' Content information of the precalculated results:
>>> scipy.io.whosmat('demo5_2.mat')
[('pxm', (8, 500),  'double'),
 ('t',   (1, 1000), 'double'),
 ('tp',  (1, 1000), 'double'),
 ('tsd', (8, 1000), 'double'),
 ('tm',  (8, 1000), 'double')]
'''
lines = axes[2].plot(x, hier_res['pxm'][1:].T, linewidth=1)
line, = axes[2].plot(x, hier_res['pxm'][0], 'r')
axes[2].legend((line, lines[1]), ('school A', 'other schools'))
axes[2].set_yticks(())
axes[2].set_title('hierarchical model')
axes[2].set_xlabel('Treatment effect')


plt.show()

