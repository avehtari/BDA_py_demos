"""Bayesian Data Analysis, 3rd ed
Chapter 3, demo 4

Visualise sampling from the posterior predictive distribution.

"""

from __future__ import division
import os, threading
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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
# sample from the predictive distribution p(ynew|y)
# for each sample of (mu, sigma)
ynew = np.random.randn(*mu.shape)*sigma + mu


# For mu compute the density in these points
tl1 = [90, 150]
t1 = np.linspace(tl1[0], tl1[1], 1000)
# For sigma compute the density in these points
tl2 = [10, 60]
t2 = np.linspace(tl2[0], tl2[1], 1000)
# For ynew compute the density in these points
tlynew = [50, 185]
xynew = np.linspace(tlynew[0], tlynew[1], 1000)

# evaluate the joint density in a grid
# note that the following is not normalized, but for plotting
# contours it does not matter
Z = stats.norm.pdf(t1, my, t2[:,np.newaxis]/np.sqrt(n))
Z *= (sinvchi2.pdf(t2**2, n-1, s2)*2*t2)[:,np.newaxis]

# compute the exact predictive density
# multiplication by 1./sqrt(s2/n) is due to the transformation of variable
# see BDA3 p. 21
p_new = stats.t.pdf((xynew-my)/np.sqrt(s2*(1+1/n)), n-1) / np.sqrt(s2*(1+1/n))


# ====== Illustrate the sampling with interactive plot

# create figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,8))

# plot the joint distribution
# plot the contour plot of the exact posterior (c_levels is used to give
# a vector of linearly spaced values at which levels contours are drawn)
c_levels = np.linspace(1e-5, Z.max(), 6)[:-1]
axes[0].contour(t1, t2, Z, c_levels, colors='blue')
# plot the samples from the joint posterior
samps = axes[0].scatter(mu, sigma, 5, color=[0.25, 0.75, 0.25])
# decorate
axes[0].set_xlim(tl1)
axes[0].set_ylim(tl2)
axes[0].set_xlabel('$\mu$', fontsize=20)
axes[0].set_ylabel('$\sigma$', fontsize=20)
axes[0].set_title('joint posterior')
axes[0].legend(
    (plt.Line2D([], [], color='blue'), samps),
    ('exact contour plot', 'samples'),
    loc='upper center'
)
# highlight the first sample
ax0_hs = axes[0].scatter(mu[0], sigma[0], 40, 'r')


# plot ynew
# precalculate each predictive pdf with given mu and sigma
ynewdists = stats.norm.pdf(xynew, mu[:,np.newaxis], sigma[:,np.newaxis])
# plot the first distribution and the respective sample
line1, = axes[1].plot(xynew, ynewdists[0], 'b', linewidth=1.5)
ax1_hs = axes[1].scatter(ynew[0], 0.02*np.max(ynewdists), 40, 'r')
# decorate
axes[1].set_ylim([0, np.max(ynewdists)])
axes[1].set_xlim(tlynew)
axes[1].set_xlabel('$\widetilde{y}$', fontsize=20)
axes[1].set_yticks(())
axes[1].legend(
    (line1, ax1_hs),
    ('pred.dist. given the posterior sample',
    'sample from the predictive distribution'),
    loc='upper center'
)

# function for interactively updating the figure
def update_figure(event):
    
    if icontainer.stage == 0:
        icontainer.stage += 1
        # modify helper text
        htext.set_text('press `q` to skip animation')
        # start the timer
        anim_thread.start()
    
    elif icontainer.stage == 1 and event.key == 'q':
        # stop the animation
        stop_anim.set()
    
    elif icontainer.stage == 2:
        icontainer.stage += 1
        # remove helper text
        htext.set_text('')
        line, = axes[1].plot(xynew, p_new, linewidth=1.5)
        # update legend
        axes[1].legend(
            (icontainer.ax1_hs, line),
            ('samples from the predictive distribution',
             'exact predictive distribution'),
             loc='upper center'
        )
        fig.canvas.draw()
        

# function for performing the figure animation in thread
def animation():
    for i1 in xrange(1, nanim):
        # remove previous highlights
        icontainer.ax0_hs.remove()
        icontainer.ax1_hs.get_sizes()[0] = 20
        icontainer.ax1_hs.get_facecolor()[0,0] = 0
        icontainer.ax1_hs.get_facecolor()[0,1] = 0.5
        icontainer.ax1_hs.get_facecolor()[0,3] = 0.2
        icontainer.ax1_hs.get_edgecolor()[0,1] = 0.5
        icontainer.ax1_hs.get_edgecolor()[0,3] = 0.2
        # remove previous predictive distribution
        axes[1].lines.remove(icontainer.prev_line)
        # show next sample
        icontainer.ax0_hs = axes[0].scatter(mu[i1], sigma[i1], 40, 'r')
        icontainer.ax1_hs = axes[1].scatter(
            ynew[i1], (0.02 + 0.02*np.random.rand())*np.max(ynewdists), 40, 'r'
        )
        icontainer.prev_line, = axes[1].plot(
            xynew, ynewdists[i1], 'b', linewidth=1.5
        )
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
    # remove highlights
    icontainer.ax0_hs.remove()
    axes[1].lines.remove(icontainer.prev_line)
    icontainer.ax1_hs.get_sizes()[0] = 20
    icontainer.ax1_hs.get_facecolor()[0,0] = 0
    icontainer.ax1_hs.get_facecolor()[0,1] = 0.5
    icontainer.ax1_hs.get_facecolor()[0,3] = 0.2
    icontainer.ax1_hs.get_edgecolor()[0,1] = 0.5
    icontainer.ax1_hs.get_edgecolor()[0,3] = 0.2
    # modify helper text
    htext.set_text('press any key to continue')
    # plot the rest of the samples
    i1 += 1
    icontainer.ax1_hs = axes[1].scatter(
        ynew[i1:], (0.02 + 0.015*np.random.rand(nsamp-i1))*np.max(ynewdists), 10,
        color=[0,0.5,0], alpha=0.2
    )
    # update legend
    axes[1].legend(
        (icontainer.ax1_hs,),
        ('samples from the predictive distribution',),
        loc='upper center'
    )
    fig.canvas.draw()

# animation related variables
stop_anim = threading.Event()
anim_thread = threading.Thread(target=animation)
anim_delay = 0.25
nanim = 50

# store the information of the current stage of the figure
class icontainer(object):
    stage = 0
    ax0_hs = ax0_hs
    ax1_hs = ax1_hs
    prev_line = line1

# add helper text
htext = fig.suptitle('press any key to continue', fontsize=20)
# set figure to react to keypress events
fig.canvas.mpl_connect('key_press_event', update_figure)
# start blocking figure
plt.show()

