"""Bayesian Data Analysis, 3rd ed
Chapter 2, demo 1

437 girls and 543 boys have been observed. Calculate and plot the posterior 
distribution of the proportion of girls $\theta$, using uniform prior on 
$\theta$.

"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


# Edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2)
plt.rc('axes', color_cycle=('#377eb8','#e41a1c','#4daf4a',
                            '#984ea3','#ff7f00','#ffff33'))

# The posterior distribution is Beta(438,544)

# Create grid of 150 points from 0.36 to 0.54
x = np.linspace(0.36, 0.54, 150)
# Freeze a beta distribution object with given parameters
dist = beta(438, 544)
# Probability density function at x
pd = dist.pdf(x)

# Plot pd
plt.plot(x, pd)

# Plot proportion of girl babies in general population as a vertical line
plt.axvline(0.485, color='#4daf4a', linewidth=1.5, alpha=0.5)

# Find the points in x that are between 2.5% and 97.5% quantile
# dist.ppf is percent point function (inverse of cdf)
x_95_idx = (x > dist.ppf(0.025)) & (x < dist.ppf(0.975))
# Shade the 95% central posterior interval
plt.fill_between(x[x_95_idx], pd[x_95_idx], color=(0.9,0.9,0.9))
# Add text into the shaded area
plt.text(dist.median(), 8, "95%", horizontalalignment='center')
# Add labels and title
plt.xlabel(r'$\theta$', fontsize=18)
plt.ylabel(r'$p(\theta|y,n)$', fontsize=18)
plt.title('Uniform prior -> Posterior is Beta(438,544)')
# Remove ticks from the y-axis
plt.yticks(())
# Scale x-axis to the data
plt.autoscale(axis='x', tight=True)

# Display the figure
plt.show()
