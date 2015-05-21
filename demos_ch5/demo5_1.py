"""Becs-114.1311 Introduction to Bayesian Statistics
Chapter 5, demo 1

Hierarchical model for Rats experiment (BDA3, p. 102).

"""

from __future__ import division
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# rat data (BDA3, p. 102)
y = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 
    2, 2, 2, 2, 2, 2, 1, 5, 2, 5, 3, 2, 7, 7, 3, 3, 2, 9, 10, 4, 4, 4, 4, 4, 4, 
    4, 10, 4, 4, 4, 5, 11, 12, 5, 5, 6, 5, 6, 6, 6, 6, 16, 15, 15, 9, 4
])    
n = np.array([
    20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 18, 18, 17, 20, 20, 20, 20, 19, 
    19, 18, 18, 25, 24, 23, 20, 20, 20, 20, 20, 20, 10, 49, 19, 46, 27, 17, 49, 
    47, 20, 20, 13, 48, 50, 20, 20, 20, 20, 20, 20, 20, 48, 19, 19, 19, 22, 46, 
    49, 20, 20, 23, 19, 22, 20, 20, 20, 52, 46, 47, 24, 14
])
M = len(y)

# plot separate and pooled model
fig, axes = plt.subplots(2, 1, figsize=(8,10))
x = np.linspace(0, 1, 200)

# separate
for i in range(M-1):
    line1, = axes[0].plot(
        x, beta.pdf(x, y[i] + 1, n[i] - y[i] + 1), color=[0.3,0.5,0.8]
    )
# emphasise last sample
line2, = axes[0].plot(
    x, beta.pdf(x, y[-1] + 1, n[-1] - y[-1] + 1), 'r', linewidth=2.5
)
axes[0].legend((line1, line2),
           ('Posterior of $\\theta_j$', 'Posterior of $\\theta_{71}$'))
axes[0].set_yticks(())
axes[0].set_xlabel('$\\theta$', fontsize=20)
axes[0].set_title('separate model')

# pooled
line1, = axes[1].plot(
    x, beta.pdf(x, y.sum() + 1, n.sum() - y.sum() + 1),
    color=[0.3,0.5,0.8], linewidth=2.5
)
axes[1].legend((line1,), ('Posterior of common $\\theta$',))
axes[1].set_yticks(())
axes[1].set_xlabel('$\\theta$', fontsize=20)
axes[1].set_title('pooled model')
