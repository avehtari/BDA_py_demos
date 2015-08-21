"""Bayesian Data Analysis, 3rd ed
Chapter 2, demo 4

Calculate the posterior distribution on a discrete grid of points by 
multiplying the likelihood and a non-conjugate prior at each point, and 
normalizing over the points. Simulate samples from the resulting non-standard 
posterior distribution using inverse cdf using the discrete grid.

"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


# Edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2)
plt.rc('axes', color_cycle=('#377eb8','#e41a1c','#4daf4a',
                            '#984ea3','#ff7f00','#ffff33'))

# Data (437,543)
a = 437
b = 543

# Grid of nx points
nx = 1000
x = np.linspace(0, 1, nx)

# Compute density of non-conjugate prior in grid
# This non-conjugate prior is same as in Figure 2.4 in the book
pp = np.ones(nx)
ascent = (0.385 <= x) & (x <= 0.485)
descent = (0.485 <= x) & (x <= 0.585)
pm = 11
pp[ascent] = np.linspace(1, pm, np.count_nonzero(ascent))
pp[descent] = np.linspace(pm, 1, np.count_nonzero(descent))
# Normalize the prior
pp /= np.sum(pp)

# Unnormalised non-conjugate posterior in grid
po = beta.pdf(x, a, b)*pp
po /= np.sum(po)
# Cumulative
pc = np.cumsum(po)

# Inverse-cdf sampling
# Get n uniform random numbers from [0,1]
n = 10000
r = np.random.rand(n)
# Map each r into corresponding grid point x:
# [0, pc[0]) map into x[0] and [pc[i-1], pc[i]), i>0, map into x[i]
rr = x[np.sum(pc[:,np.newaxis] < r, axis=0)]


# Plot posteriors
# Plot 3 subplots
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
# Posterior with uniform prior Beta(1,1)
axes[0].plot(x, beta.pdf(x, a+1, b+1))
axes[0].set_title('Poster with uniform prior')
axes[0].set_yticks(())
# Non-conjugate prior
axes[1].plot(x, pp)
axes[1].set_title('Non-conjugate prior')
axes[1].set_yticks(())
# Posterior with non-conjugate prior
axes[2].plot(x, po)
axes[2].set_title('Posterior with non-conjugate prior')
axes[2].set_yticks(())
# Set custom limits for x-axis
axes[0].set_xlim((0.35, 0.6))
fig.subplots_adjust(hspace=0.2)

# Plot samples
fig = plt.figure()
# Plot cumulative posterior
plt.plot(x, pc, color='#e41a1c')
# Calculate histograms and scale them into the same figure
hist_r = np.histogram(r, bins=30)
hist_rr = np.histogram(rr, bins=30)
plt.barh(hist_r[1][:-1], hist_r[0]*0.02/hist_r[0].max(),
        height=hist_r[1][1]-hist_r[1][0], left=0.35, color='#4daf4a')
plt.bar(hist_rr[1][:-1], hist_rr[0]*0.2/hist_rr[0].max(),
        width=hist_rr[1][1]-hist_rr[1][0], color='#377eb8')
plt.legend(('Cumulative posterior', 'Random uniform numbers',
            'Posterior samples'), loc='best')
# Set limits
plt.xlim((0.35, 0.55))
plt.ylim((0,1))

# Display the figure
plt.show()
