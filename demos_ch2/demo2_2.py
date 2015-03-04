"""Becs-114.1311 Introduction to Bayesian Statistics
Chapter 2, demo 2

Illustrate the effect of prior. Comparison of posterior distributions with 
different parameter values for the beta prior distribution.

"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


# Plotting grid
x = np.linspace(0.375, 0.525, 100)

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

# Plot 3 subplots
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True,
                         figsize=(8, 12))
for i in range(3):
    # Plot three precalculated densities
    axes[i].plot(x, pdu)
    axes[i].plot(x, pdp[i], 'k:')
    axes[i].plot(x, pdi[i], 'r')
    # Add vertical line
    axes[i].axvline(0.485, color='k', linestyle='--')
    # Set the title for this subplot
    axes[i].set_title('alpha/(alpha+beta) = 0.485, alpha+beta = {}'
                      .format(2*10**i))
    # Limit xaxis
    axes[i].autoscale(axis='x', tight=True)
# Add legend to the first subplot
axes[0].legend(('Post with unif prior', 'Informative prior', 'Posterior'))

# Display the figure
plt.show()
