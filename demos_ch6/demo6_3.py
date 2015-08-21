"""Bayesian Data Analysis, 3rd ed
Chapter 6, demo 3

Posterior predictive checking
Light speed example with a poorly chosen test statistic

"""

from __future__ import division
import numpy as np
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
s2 = np.var(y, ddof=1)  # Here ddof=1 is used to get the sample estimate.
my = np.mean(y)

# A second example of replications
nsamp = 1000
pps = np.random.standard_t(n-1, size=(n,nsamp))*np.sqrt(s2*(1+1/n)) + my
# Use the sample variance as a test statistic
# This is a poor choice since it corresponds directly to
# the variance parameter in the model which has been fitted
# to the data.
pp = np.var(pps, axis=0, ddof=1)

# ====== plot
plt.hist(pp, 20, label='Variances of the replicated data sets')
plt.axvline(s2, color='#e41a1c', label='Variance of the original data')
plt.yticks(())
plt.title('Light speed example with a poorly chosen test statistic\n'
    r'$\operatorname{Pr}(T(y_\mathrm{rep},\theta)\leq T(y,\theta)|y)=0.42$')
plt.legend()
# make room for the title and legend
axis = plt.gca()
axis.set_ylim((0, axis.get_ylim()[1]*1.2))
box = axis.get_position()
axis.set_position([box.x0, box.y0, box.width, box.height * 0.9])

plt.show()

