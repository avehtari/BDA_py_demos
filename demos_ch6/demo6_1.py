"""Bayesian Data Analysis, 3rd ed
Chapter 6, demo 1

Posterior predictive checking demo

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
s2 = np.var(y, ddof=1)  # Here ddof=1 is used to get the sample estimate.
s = np.sqrt(s2)
my = np.mean(y)

# Create 9 random replicate data sets from the posterior predictive density.
# Each set has same number of virtual observations as the original data set.
replicates = np.random.standard_t(n-1, size=(9,n)) * np.sqrt(1+1/n)*s + my

# plot them along with the real data set in random order subplot
fig, axes = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(10,12))
order = np.random.permutation(10)
for i in range(10):
    a = axes.ravel()[i]
    a.hist(replicates[order[i]] if order[i] < 9 else y, np.arange(-45,55,5))
    a.set_yticks(())
axes[0,0].set_xlim([-48,58])
fig.suptitle('Light speed example: Observed data + Replicated datasets.\n'
             'Can you spot which one is the observed data?', fontsize=18)

"""
The distribution of the minimum value of a replicated data set can be 
calculated analytically. Consider n samples of X_i, where X_i has cumulative 
distribution function F(x) and probability distributoin function f(x). The 
cumulative distribution function of the minimum of the n samples is    
    1 - (1 - F(x))^n
and the probability distribution function is its derivative
    n * f(x) * (1 - F(x))^(n-1).

"""

# Calculate the pdf of the minumum of a replicated dataset
x = np.linspace(-60, 20, 150)
pdf = stats.t.pdf(x, df=n-1, loc=my, scale=np.sqrt(s2*(1+1/n)))
cdf = stats.t.cdf(x, df=n-1, loc=my, scale=np.sqrt(s2*(1+1/n)))
pdf_min = n * pdf * (1 - cdf)**(n-1)

# Plot the real minimum and the distribution of the min of a replicate data set
plt.figure()
plt.plot(x, pdf_min,
         label='distribution of the minimum of a replicated data set')
plt.yticks(())
plt.axvline(y.min(), color='k', linestyle='--',
            label='minimum of the true data set')
plt.ylim([0,0.11])
plt.legend(loc='upper center')

plt.show()

