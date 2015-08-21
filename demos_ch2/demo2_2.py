"""Bayesian data analysis, 3rd ed
Chapter 2, demo 2

Illustrate the effect of a prior. Comparison of posterior distributions with 
different parameter values for Beta prior distribution.

"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


# Edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2)
plt.rc('axes', color_cycle=('#377eb8','#e41a1c','#4daf4a',
                            '#984ea3','#ff7f00','#ffff33'))

# Grid
x = np.linspace(0.36, 0.54, 150)

# Posterior with data (437,543) and uniform prior Beta(1,1)
au = 438
bu = 544
# Calculate densities
pdu = beta.pdf(x, au, bu)

# Compare 3 cases
# Arrays of different priors: Beta(0.485*n,(1-0.485)*n), for n = 2, 20, 200
ap = np.array([0.485 * (2*10**i) for i in range(3)])
bp = np.array([(1-0.485) * (2*10**i) for i in range(3)])
# Corresponding posteriors with data (437,543)
ai = 437 + ap
bi = 543 + bp
# Calculate prior and posterior densities
pdp = beta.pdf(x, ap[:,np.newaxis], bp[:,np.newaxis])
pdi = beta.pdf(x, ai[:,np.newaxis], bi[:,np.newaxis])

"""
The above two expressions uses numpy broadcasting inside the `beta.pdf` 
function. Arrays `ap` and `bp` have shape (3,) i.e. they are 1d arrays of 
length 3. Array `x` has shape (150,) and the output `pdp` is an array of shape 
(3,150).

Instead of using the `beta.pdf` function, we could have also calculated other 
arithmetics. For example `out = x + (ap * bp)[:,np.newaxis]` returns an array 
of shape (3,150), where each element `out[i,j] = x[j] + ap[i] * bp[i]`.

With broadcasting, unnecessary repetition is avoided, i.e. it is not necessary 
to create an array of `ap` repeated 150 times into the memory. More info can be 
found on the numpy documentation. Compare to `bsxfun` in Matlab.

"""

# Plot 3 subplots
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True,
                         figsize=(8, 12))
# Leave space for the legend on bottom and remove some space from the top
fig.subplots_adjust(bottom=0.2, top=0.94)
for i in range(3):
    # Add vertical line
    known = axes[i].axvline(0.485, color='#4daf4a', linewidth=1.5, alpha=0.5)
    # Plot three precalculated densities
    post1, = axes[i].plot(x, pdu, color='#ff8f20', linewidth=2.5)
    prior, = axes[i].plot(x, pdp[i], 'k:', linewidth=1.5)
    post2, = axes[i].plot(x, pdi[i], 'k--', linewidth=1.5)
    plt.yticks(())
    # Set the title for this subplot
    axes[i].set_title(r'$\alpha/(\alpha+\beta) = 0.485,\quad \alpha+\beta = {}$'
                      .format(2*10**i), fontsize=18)
# Limit xaxis
axes[0].autoscale(axis='x', tight=True)
axes[0].set_ylim((0,30))
# Add legend to the last subplot
axes[-1].legend(
    (post1, prior, post2, known),
    ( 'posterior with uniform prior',
      'informative prior',
      'posterior with informative prior',
     r'known $\theta=0.485$ in general population'),
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15)
)

# Display the figure
plt.show()
