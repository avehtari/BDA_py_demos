"""Bayesian Data Analysis, 3rd ed
Chapter 3, demo 2

Visualise factored sampling and the corresponding marginal and conditional densities.

"""

from __future__ import division
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# import from utilities
import os
util_path = '../utilities_and_data'  # provide path to utilities
util_path = os.path.abspath(util_path)
if util_path not in os.sys.path and os.path.exists(util_path):
    os.sys.path.insert(0, util_path)
import sinvchi2


# Edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)

# data
y = np.array([93, 112, 122, 135, 122, 150, 118, 90, 124, 114])
# sufficient statistics
n = len(y)
s2 = np.var(y, ddof=1)  # Here ddof=1 is used to get the sample estimate.
my = np.mean(y)

# Factorize the joint posterior p(mu,sigma2|y) to p(sigma2|y)p(mu|sigma2,y)
# Sample from the joint posterior using this factorization

# sample from p(sigma2|y)
nsamp = 1000
sigma2 = sinvchi2.rvs(n-1, s2, size=nsamp)
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

# evaluate the joint density in grid
# note that the following is not normalized, but for plotting
# contours it does not matter
Z = stats.norm.pdf(t1, my, t2[:,np.newaxis]/np.sqrt(n))
Z *= (sinvchi2.pdf(t2**2, n-1, s2)*2*t2)[:,np.newaxis]

# compute the exact marginal density for sigma
# multiplication by 2*t2 is due to the transformation of variable
# z=t2^2, see BDA3 p. 21
pm_sigma = sinvchi2.pdf(t2**2, n-1, s2)*2*t2
# N.B. this was already calculated in the joint distribution case


# ====== Illustrate the sampling with interactive plot

# create figure
plotgrid = gridspec.GridSpec(1, 2, width_ratios=[3,2])
fig = plt.figure(figsize=(12,8))

# plot the joint distribution
ax0 = plt.subplot(plotgrid[0,0])
# plot the contour plot of the exact posterior (c_levels is used to give
# a vector of linearly spaced values at which levels contours are drawn)
c_levels = np.linspace(1e-5, Z.max(), 6)[:-1]
plt.contour(t1, t2, Z, c_levels, colors='blue')
# decorate
plt.xlim(tl1)
plt.ylim(tl2)
plt.xlabel('$\mu$', fontsize=20)
plt.ylabel('$\sigma$', fontsize=20)
plt.title('joint posterior')
plt.legend((plt.Line2D([], [], color='blue'),), ('exact contour plot',))

# plot the marginal of sigma
ax1 = plt.subplot(plotgrid[0,1])
plt.plot(pm_sigma, t2, 'b', linewidth=1.5)
# decorate
plt.ylim(tl2)
plt.title('marginal of $\sigma$')
plt.xticks(())


# Function for interactively updating the figure
def update_figure(event):
    
    if icontainer.stage == 0:
        icontainer.stage += 1
        # first sample of sigma2
        line, = ax0.plot(tl1, [sigma[0], sigma[0]], 'k--', linewidth=1.5)
        icontainer.legend_h.append(line)
        icontainer.legend_s.append('sample from the marginal of $\sigma$')
        icontainer.prev_line1 = line
        ax0.legend(icontainer.legend_h, icontainer.legend_s)
        fig.canvas.draw()
    
    elif icontainer.stage == 1:
        icontainer.stage += 1
        # the conditional distribution of mu given sigma2
        line, = ax0.plot(
            t1,
            sigma[0] + stats.norm.pdf(t1, my, np.sqrt(sigma2[0]/n))*100,
            'g--',
            linewidth=1.5
        )
        icontainer.legend_h.append(line)
        icontainer.legend_s.append('conditional distribution of $\mu$')
        icontainer.prev_line2 = line
        ax0.legend(icontainer.legend_h, icontainer.legend_s)
        fig.canvas.draw()
    
    elif icontainer.stage == 2:
        icontainer.stage += 1
        # sample mu given sigma2
        scat = ax0.scatter(mu[0], sigma[0], 40, color='g')
        icontainer.legend_h.append(scat)
        icontainer.legend_s.append('sample from joint posterior')
        icontainer.prev_scat = scat
        ax0.legend(icontainer.legend_h, icontainer.legend_s)
        fig.canvas.draw()
    
    elif icontainer.stage == 3:
        # remove the previous lines
        ax0.lines.remove(icontainer.prev_line1)
        ax0.lines.remove(icontainer.prev_line2)
        # resize the last scatter sample
        icontainer.prev_scat.get_sizes()[0] = 8
        # draw next sample
        icontainer.i1 += 1
        i1 = icontainer.i1
        # first sample of sigma2
        icontainer.prev_line1, = ax0.plot(
            tl1, [sigma[i1], sigma[i1]], 'k--', linewidth=1.5
        )
        # the conditional distribution of mu given sigma2
        icontainer.prev_line2, = ax0.plot(
            t1,
            sigma[i1] + stats.norm.pdf(t1, my, np.sqrt(sigma2[i1]/n))*100,
            'g--',
            linewidth=1.5
        )
        # sample mu given sigma2
        icontainer.prev_scat = ax0.scatter(mu[i1], sigma[i1], 40, color='g')
        # check if the last sample
        if icontainer.i1 == icontainer.ndraw-1:
            icontainer.stage += 1
        fig.canvas.draw()
    
    elif icontainer.stage == 4:
        icontainer.stage += 1
        # remove the previous lines
        ax0.lines.remove(icontainer.prev_line1)
        ax0.lines.remove(icontainer.prev_line2)
        # resize the last scatter sample
        icontainer.prev_scat.get_sizes()[0] = 8
        # remove the helper text
        plt.suptitle('')
        # remove the extra legend entries
        icontainer.legend_h.pop(2)
        icontainer.legend_h.pop(1)
        icontainer.legend_s.pop(2)
        icontainer.legend_s.pop(1)
        ax0.legend(icontainer.legend_h, icontainer.legend_s)
        # plot the remaining samples
        icontainer.i1 += 1
        i1 = icontainer.i1
        ax0.scatter(mu[i1:], sigma[i1:], 8, color='g')
        fig.canvas.draw()


# Store the information of the current stage of the figure
class icontainer(object):
    stage = 0
    i1 = 0
    legend_h = [plt.Line2D([], [], color='blue'),]
    legend_s = ['exact contour plot',]
    prev_line1 = None
    prev_line2 = None
    prev_scat = None
    ndraw = 6

plt.suptitle('Press any key to continue', fontsize=20)
fig.canvas.mpl_connect('key_press_event', update_figure)
plt.show()

