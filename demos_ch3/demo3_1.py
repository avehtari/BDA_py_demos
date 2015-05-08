"""Becs-114.1311 Introduction to Bayesian Statistics
Chapter 3, demo 1

Visualise joint density and marginal densities of posterior of normal 
distribution with unknown mean and variance.

"""

from __future__ import division
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add utilities directory to sys.path
util_path = ''  # Provide custom path to utilities ...
if not util_path:
    # ... or search it from the parent directory
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.abspath(os.path.join(cur_path, os.pardir))
    util_path = os.path.join(parent_path, 'utilities')
if util_path not in os.sys.path and os.path.exists(util_path):
    os.sys.path.insert(0, util_path)

# Import from utilities
import sinvchi2


# data
y = np.array([93, 112, 122, 135, 122, 150, 118, 90, 124, 114])
# sufficient statistics
n = len(y)
s2 = np.var(y, ddof=1)  # Here ddof=1 is used to get the sample estimate.
my = np.mean(y)

# Factorize the joint posterior p(mu,sigma2|y) to p(sigma2|y)p(mu|sigma2,y)
# Sample from the joint posterior using this factorization

# sample from p(sigma2|y) (sigma2  is a vector 1000 x 1)
sigma2 = sinvchi2.rvs(n-1, s2, size=1000)
# sample from p(mu|sigma2,y) 
mu = my + np.sqrt(sigma2/n)*np.random.randn(*sigma2.shape)
# display sigma instead of sigma2
sigma = np.sqrt(sigma2)

# For mu compute the density in these points
tl1 = [90, 150]
t1 = np.linspace(tl1[0], tl1[1], 1000)
# For sigma compute the density in these points
tl2 = [10, 60]
t2 = np.linspace(tl2[0], tl2[1], 1000)

# evaluate joint density in grid
# note that the following is not normalized, but for plotting
# contours it does not matter
Z = stats.norm.pdf(t1, my, t2[:,np.newaxis]/np.sqrt(n))
Z *= (sinvchi2.pdf(t2**2, n-1, s2)*2*t2)[:,np.newaxis]

# compute the exact marginal density for mu
# multiplication by 1./sqrt(s2/n) is due to the transformation of variable
# z=(x-mean(y))/sqrt(s2/n), see BDA3 p. 21
pm_mu = stats.t.pdf((t1 - my) / np.sqrt(s2/n), n-1) / np.sqrt(s2/n)

# estimate the marginal density for mu using samples and ad hoc Gaussian
# kernel approximation
pk_mu = stats.gaussian_kde(mu).evaluate(t1)

# compute the exact marginal density for sigma
# multiplication by 2*t2 is due to the transformation of variable
# z=t2^2, see BDA3 p. 21
pm_sigma = sinvchi2.pdf(t2**2, n-1, s2)*2*t2
# N.B. this was already calculated in the joint distribution case

# estimate the marginal density for sigma using samples and ad hoc Gaussian
# kernel approximation
pk_sigma = stats.gaussian_kde(sigma).evaluate(t2)


# ====== Plotting

# create figure
plotgrid = gridspec.GridSpec(2, 2, width_ratios=[3,2], height_ratios=[3,2])
plt.figure(figsize=(14,12))

# plot joint distribution
plt.subplot(plotgrid[0,0])
# plot the contour plot of the exact posterior (c_levels is used to give
# a vector of linearly spaced values at which levels contours are drawn)
c_levels = np.linspace(1e-5, Z.max(), 6)[:-1]
plt.contour(t1, t2, Z, c_levels, colors='blue')
# plot samples from the joint posterior
samps = plt.scatter(mu, sigma, 5, color=[0.25, 0.75, 0.25])
# decorate
plt.xlim(tl1)
plt.ylim(tl2)
plt.xlabel('$\mu$', fontsize=20)
plt.ylabel('$\sigma$', fontsize=20)
plt.title('joint posterior')
plt.legend(
    (plt.Line2D([], [], color='blue'), samps),
    ('exact contour plot', 'samples')
)

# plot marginal of mu
plt.subplot(plotgrid[1,0])
# exact
plt.plot(t1, pm_mu, 'b', label='exact')
# empirical
plt.plot(t1, pk_mu, 'r--', label='empirical')
# decorate
plt.xlim(tl1)
plt.title('marginal of $\mu$')
plt.yticks(())
plt.legend()

# plot marginal of sigma
plt.subplot(plotgrid[0,1])
# exact
plt.plot(pm_sigma, t2, 'b', label='exact')
# empirical
plt.plot(pk_sigma, t2, 'r--', label='empirical')
# decorate
plt.ylim(tl2)
plt.title('marginal of $\sigma$')
plt.xticks(())
plt.legend()

plt.show()

