"""Bayesian Data Analysis, 3rd ed
Chapter 6, demo 2

Posterior predictive checking
Binomial example - Testing sequential dependence example

"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2)
plt.rc('axes', color_cycle=('#377eb8','#e41a1c','#4daf4a',
                            '#984ea3','#ff7f00','#ffff33'))

# Testing sequential dependence example (Gelman et al p. 163)
y = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
Ty = np.count_nonzero(np.diff(y))

# sufficient statistics
n = len(y)
s = y.sum()

nsamp = 10000
t = np.random.beta(s+1, n-s+1, size=nsamp)
yr = np.random.rand(n, nsamp) < t
# sadly np.count_nonzero does not (yet) support axis parameter
Tyr = (np.diff(yr, axis=0) != 0).sum(axis=0)

# ====== plot
plt.hist(Tyr, np.arange(19), align='left', label='$T(y_\mathrm{rep})$')
plt.axvline(Ty, color='#e41a1c', label='$T(y)$')
plt.yticks(())
plt.xlim((-0.5,17.5))
plt.title('Binomial example - number of changes? \n'
    r'$\operatorname{Pr}(T(y_\mathrm{rep},\theta) \leq T(y,\theta)|y) = 0.03$')
plt.legend()
# make room for the title
axis = plt.gca()
box = axis.get_position()
axis.set_position([box.x0, box.y0, box.width, box.height * 0.9])

plt.show()

