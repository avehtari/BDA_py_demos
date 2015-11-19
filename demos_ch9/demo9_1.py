"""Bayesian Data Analysis, 3rd ed
Chapter 9, demo 1

Prof Gelman has a jar of coins. He promises that if the students
guess how many coins there are, they will get all the coins in
the jar. Students discuss and guess different values. Based on these
they eventually present their uncertainty about the number of coins as
a normal distribution N(160,40). What value they should guess?

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

# distribution
m = 160     # mean
s = 40      # std

x = np.arange(m-3*s, m+3*s+1)
px = stats.norm.pdf(x, loc=m, scale=s)
xpx = x * px

fig, axes = plt.subplots(3, 1, sharex=True)
axes[-1].set_xlabel('Number of coins')
axes[-1].set_xlim([x[0], x[-1]])

axes[0].plot(x, px)
axes[0].set_ylim([0, 1.3*np.max(px)])
axes[0].set_yticks(())
axes[0].set_ylabel('Probability')

# If students just want to guess right, and they do not care how much money
# they'll get they should guess the most probable value.
h2, = axes[0].plot([m, m], [0, stats.norm.pdf(m, loc=m, scale=s)])
axes[0].legend((h2,), ('Most probable value = {}'.format(m),))

# Alternatively students might want to maximize the exepected utility of the
# number coins. Assume that utility of the money is linear.
# Plot the utility 
axes[1].plot(x, x)
axes[1].set_ylabel('Utility if guess is correct')

# If students guess value a, given their estimate of the uncertainity,
# probability that they get a coins is p(a), and expected utility is a*p(a).
# Plot the expected utility 
axes[2].plot(x, xpx)
axes[2].set_ylabel('Expected utility')
axes[2].set_ylim([0, 1.3*np.max(xpx)])

# Compute the maximum of the expected utility
mi = np.argmax(xpx)
meu = xpx[mi]
meux = x[mi]

h3, = axes[2].plot([meux, meux], [0, meu])
axes[2].legend(
    (h3,),
    ('The guess which maximises the expected utility = {}'.format(meux),)
)

plt.show()

