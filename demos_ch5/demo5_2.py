"""Bayesian Data Analysis, 3rd ed
Chapter 5, demo 2

Hierarchical model for SAT-example data (BDA3, p. 102)

"""

from __future__ import division
import numpy as np
from scipy.stats import norm
import scipy.io # For importing a matlab file
import matplotlib.pyplot as plt

# Edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2)
plt.rc('axes', color_cycle=(plt.rcParams['lines.color'],)) # Disable color cycle

# SAT-example data (BDA3 p. 120)
# y is the estimated treatment effect
# s is the standard error of effect estimate
y = np.array([28,  8, -3,  7, -1,  1, 18, 12])
s = np.array([15, 10, 16, 11,  9, 11, 10, 18])
M = len(y)

# load the pre-computed results for the hierarchical model
# replace this with your own code in Ex 5.1*
hres_path = '../utilities_and_data/demo5_2.mat'
hres = scipy.io.loadmat(hres_path)
''' Content information of the precalculated results:
>>> scipy.io.whosmat('demo5_2.mat')
[('pxm', (8, 500),  'double'),
 ('t',   (1, 1000), 'double'),
 ('tp',  (1, 1000), 'double'),
 ('tsd', (8, 1000), 'double'),
 ('tm',  (8, 1000), 'double')]
'''
pxm = hres['pxm']
t   = hres['t'][0]
tp  = hres['tp'][0]
tsd = hres['tsd']
tm  = hres['tm']


# plot the separate, pooled and hierarchical models
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8,10))
x = np.linspace(-40, 60, 500)

# separate
lines = axes[0].plot(x, norm.pdf(x[:,None], y[1:], s[1:]), linewidth=1)
line, = axes[0].plot(x, norm.pdf(x, y[0], s[0]), 'r')
axes[0].legend((line, lines[1]), ('school A', 'other schools'),
               loc='upper left')
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
    label='All schools'
)
axes[1].legend(loc='upper left')
axes[1].set_yticks(())
axes[1].set_title('pooled model')

# hierarchical
lines = axes[2].plot(x, pxm[1:].T, linewidth=1)
line, = axes[2].plot(x, pxm[0], 'r')
axes[2].legend((line, lines[1]), ('school A', 'other schools'),
               loc='upper left')
axes[2].set_yticks(())
axes[2].set_title('hierarchical model')
axes[2].set_xlabel('Treatment effect')


# plot various marginal and conditional posterior summaries
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8,10))

axes[0].plot(t, tp)
axes[0].set_yticks(())
axes[0].set_title(r'marginal posterior density $p(\tau|y)$')
axes[0].set_ylabel(r'$p(\tau|y)$', fontsize=20)
axes[0].set_xlim([0,35])

lines = axes[1].plot(t, tm[1:].T, linewidth=1)
line, = axes[1].plot(t, tm[0].T, 'r')
axes[1].legend((line, lines[1]), ('school A', 'other schools'),
               loc='upper left')
axes[1].set_title(r'conditional posterior means of effects '
                  r'$\operatorname{E}(\theta_j|\tau,y)$')
axes[1].set_ylabel(r'$\operatorname{E}(\theta_j|\tau,y)$', fontsize=20)

lines = axes[2].plot(t, tsd[1:].T, linewidth=1)
line, = axes[2].plot(t, tsd[0].T, 'r')
axes[2].legend((line, lines[1]), ('school A', 'other schools'),
               loc='upper left')
axes[2].set_title(r'standard deviations of effects '
                  r'$\operatorname{sd}(\theta_j|\tau,y)$')
axes[2].set_ylabel(r'$\operatorname{sd}(\theta_j|\tau,y)$', fontsize=20)
axes[2].set_xlabel(r'$\tau$', fontsize=20)

plt.show()

