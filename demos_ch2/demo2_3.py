"""Bayesian Data Analysis, 3rd ed
Chapter 2, demo 3

Simulate samples from Beta(438,544), draw a histogram with quantiles, and do 
the same for a transformed variable.

"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


# Edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2)
plt.rc('axes', color_cycle=('#377eb8','#e41a1c','#4daf4a',
                            '#984ea3','#ff7f00','#ffff33'))

# Plotting grid
x = np.linspace(0.36, 0.54, 150)

# Draw n random samples from Beta(438,544)
n = 10000
th = beta.rvs(438, 544, size=n)  # rvs comes from `random variates`

# Plot 2 subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

# Plot histogram
axes[0].hist(th, bins=30)
# Compute 2.5% and 97.5% quantile approximation using samples
th25, th975 = np.percentile(th, [2.5, 97.5])
# Draw lines for these
axes[0].axvline(th25, color='#e41a1c', linewidth=1.5)
axes[0].axvline(th975, color='#e41a1c', linewidth=1.5)
axes[0].text(th25, axes[0].get_ylim()[1]+15, '2.5%',
             horizontalalignment='center')
axes[0].text(th975, axes[0].get_ylim()[1]+15, '97.5%',
             horizontalalignment='center')
axes[0].set_xlabel(r'$\theta$', fontsize=18)
axes[0].set_yticks(())

# Plot histogram for the transformed variable
phi = (1-th)/th
axes[1].hist(phi, bins=30)
# Compute 2.5% and 97.5% quantile approximation using samples
phi25, phi975 = np.percentile(phi, [2.5, 97.5])
# Draw lines for these
axes[1].axvline(phi25, color='#e41a1c', linewidth=1.5)
axes[1].axvline(phi975, color='#e41a1c', linewidth=1.5)
axes[1].text(phi25, axes[1].get_ylim()[1]+15, '2.5%',
             horizontalalignment='center')
axes[1].text(phi975, axes[1].get_ylim()[1]+15, '97.5%',
             horizontalalignment='center')
axes[1].set_xlabel(r'$\phi$', fontsize=18)
axes[1].set_yticks(())

# Display the figure
plt.show()
