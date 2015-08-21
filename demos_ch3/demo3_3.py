"""Bayesian Data Analysis, 3rd ed
Chapter 3, demo 3

Visualise the marginal distribution of mu as a mixture of normals.

"""

from __future__ import division
import os, threading
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

# evaluate the joint density in a grid
# note that the following is not normalized, but for plotting
# contours it does not matter
Z = stats.norm.pdf(t1, my, t2[:,np.newaxis]/np.sqrt(n))
Z *= (sinvchi2.pdf(t2**2, n-1, s2)*2*t2)[:,np.newaxis]

# compute the exact marginal density for mu
# multiplication by 1./sqrt(s2/n) is due to the transformation of variable
# z=(x-mean(y))/sqrt(s2/n), see BDA3 p. 21
pm_mu = stats.t.pdf((t1 - my) / np.sqrt(s2/n), n-1) / np.sqrt(s2/n)

# compute the exact marginal density for sigma
# multiplication by 2*t2 is due to the transformation of variable
# z=t2^2, see BDA3 p. 21
pm_sigma = sinvchi2.pdf(t2**2, n-1, s2)*2*t2
# N.B. this was already calculated in the joint distribution case


# ====== Illustrate the sampling with interactive plot

# create figure
plotgrid = gridspec.GridSpec(2, 2, width_ratios=[3,2], height_ratios=[3,2])
fig = plt.figure(figsize=(14,12))

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

# plot marginal of sigma
ax1 = plt.subplot(plotgrid[0,1])
plt.plot(pm_sigma, t2, 'b', linewidth=1.5)
# decorate
plt.ylim(tl2)
plt.title('marginal of $\sigma$')
plt.xticks(())

# plot the marginal of mu (empty first)
ax10 = plt.subplot(plotgrid[1,0])
plt.xlim(tl1)
plt.title('marginal of $\mu$')
plt.yticks(())

# precalculate the conditional pdfs for each sample
condpdfs = stats.norm.pdf(t1, my, np.sqrt(sigma2/n)[:,np.newaxis])

# function for interactively updating the figure
def update_figure(event):
    
    if icontainer.stage == 0:
        icontainer.stage = 3
        # first sample of sigma2
        line1, = ax0.plot(tl1, [sigma[0], sigma[0]], 'k--', linewidth=1.5)
        icontainer.legend_h.append(line1)
        icontainer.legend_s.append('sample from the marginal of $\sigma$')
        icontainer.prev_line1 = line1
        # the conditional distribution of mu given sigma2
        line2, = ax0.plot(t1, sigma[0] + condpdfs[0]*100, 'g--', linewidth=1.5)
        icontainer.legend_h.append(line2)
        icontainer.legend_s.append('conditional distribution of $\mu$')
        icontainer.prev_line2 = line2
        conddist, = ax10.plot(t1, condpdfs[0], 'g--')
        # sample mu given sigma2
        scat = ax0.scatter(mu[0], sigma[0], 40, color='g')
        icontainer.legend_h.append(scat)
        icontainer.legend_s.append('sample from joint posterior')
        icontainer.prev_scat = scat
        ax0.legend(icontainer.legend_h, icontainer.legend_s)
        fig.canvas.draw()
    
    elif icontainer.stage == 3:
        icontainer.stage += 1
        # modify helper text
        htext.set_text('press `q` to skip animation')
        # start the timer
        anim_thread.start()
    
    elif icontainer.stage == 4 and event.key == 'q':
        # stop the animation
        stop_anim.set()
    
    elif icontainer.stage == 5:
        icontainer.stage += 1
        # plot rest of the samples
        ax0.scatter(mu[nanim:], sigma[nanim:], 8, color='g')
        ax10.clear()
        ax10.set_ylim([0,0.09])
        ax10.set_xlim(tl1)
        ax10.set_title('marginal of $\mu$')
        ax10.plot(t1, np.mean(condpdfs, axis=0),
                  color='#ff8f20', linewidth=4,
                  label='average of sampled conditionals')
        ax10.plot(t1, pm_mu, 'k--', linewidth=1.5, label='exact')
        ax10.legend()
        ax10.set_yticks(())
        htext.set_text('')
        fig.canvas.draw()

# function for performing the figure animation in thread
def animation():
    for i1 in xrange(1, nanim):
        # remove previous lines
        ax0.lines.remove(icontainer.prev_line1)
        ax0.lines.remove(icontainer.prev_line2)
        # resize last scatter sample
        icontainer.prev_scat.get_sizes()[0] = 8
        # draw next sample
        # first sample of sigma2
        icontainer.prev_line1, = \
            ax0.plot(tl1, [sigma[i1], sigma[i1]], 'k--', linewidth=1.5)
        # the conditional distribution of mu given sigma2
        icontainer.prev_line2, = ax0.plot(
            t1, sigma[i1] + condpdfs[i1]*100,
            'g--', linewidth=1.5
        )
        conddist, = ax10.plot(t1, condpdfs[i1], 'g--')
        # sample mu given sigma2
        icontainer.prev_scat = ax0.scatter(mu[i1], sigma[i1], 40, color='g')
        # update figure
        fig.canvas.draw()
        # wait animation delay time or until animation is cancelled
        stop_anim.wait(anim_delay)
        if stop_anim.isSet():
            # animation cancelled
            break
    # skip the rest if the figure does not exist anymore
    if not plt.fignum_exists(fig.number):
        return    
    # advance stage
    icontainer.stage += 1
    # remove previous lines
    ax0.lines.remove(icontainer.prev_line1)
    ax0.lines.remove(icontainer.prev_line2)
    # resize last scatter sample
    icontainer.prev_scat.get_sizes()[0] = 8
    # remove helper text
    htext.set_text('press any key to continue')
    # remove extra legend entries
    icontainer.legend_h.pop(2)
    icontainer.legend_h.pop(1)
    icontainer.legend_s.pop(2)
    icontainer.legend_s.pop(1)
    ax0.legend(icontainer.legend_h, icontainer.legend_s)
    # plot the rest of the samples
    i1 += 1
    if i1 < nanim:
        ax0.scatter(mu[i1:nanim], sigma[i1:nanim], 8, color='g')
        conddistlist = ax10.plot(t1, condpdfs[i1:nanim].T, 'g--')
    fig.canvas.draw()

# animation related variables
stop_anim = threading.Event()
anim_thread = threading.Thread(target=animation)
anim_delay = 0.25
nanim = 50

# store the information of the current stage of the figure
class icontainer(object):
    stage = 0
    legend_h = [plt.Line2D([], [], color='blue'),]
    legend_s = ['exact contour plot',]
    prev_line1 = None
    prev_line2 = None
    prev_scat = None

# add helper text
htext = fig.suptitle('press any key to continue', fontsize=20)
# set figure to react to keypress events
fig.canvas.mpl_connect('key_press_event', update_figure)
# start blocking figure
plt.show()
# stop the animation (if it is active) as the figure is now closed
stop_anim.set()
