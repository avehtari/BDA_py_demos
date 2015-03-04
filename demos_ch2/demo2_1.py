"""Becs-114.1311 Introduction to Bayesian Statistics
Chapter 2, demo 1

437 girls and 543 boys have been observed. Calculate and plot the posterior distribution of the proportion of girls $\theta$, using uniform prior on $\theta$.

"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


# The posterior distribution is Beta(438,544)

# Create grid of 100 points from 0.375 to 0.525
x = np.linspace(0.375, 0.525, 100)
# Freeze a beta distribution object with given parameters
dist = beta(438, 544)
# Probability density function at x
pd = dist.pdf(x)

# Plot pd
plt.plot(x, pd)

# Plot proportion of girl babies in general population
plt.axvline(0.485, color='k', linestyle='--')

# Find the points in x that are between 2.5% and 97.5% quantile
# dist.ppf is percent point function (inverse of cdf)
x_95_idx = (x > dist.ppf(0.025)) & (x < dist.ppf(0.975))
# Shade the 95% central interval
plt.fill_between(x[x_95_idx], pd[x_95_idx], color=[0.9,0.9,0.9])
# Add text into the shaded area
plt.text(dist.median(), 8, "95%", horizontalalignment='center')

# Display the figure
plt.show()
