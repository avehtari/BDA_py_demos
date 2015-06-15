"""Bayesian data analysis
Chapter 10, demo 1

Rejection sampling example

"""

from __future__ import division
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

# edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2, markeredgewidth=0)
plt.rc('axes', color_cycle=('#377eb8','#e41a1c','#4daf4a',
                            '#984ea3','#ff7f00','#ffff33'))
plt.rc('patch', facecolor='#bfe2ff')

# fake interesting distribution
x = np.linspace(-3, 3, 200)
r = np.array([ 1.1 ,  1.3 , -0.1 , -0.7 ,  0.2 , -0.4 ,  0.06, -1.7 ,
               1.7 ,  0.3 ,  0.7 ,  1.6 , -2.06, -0.74,  0.2 ,  0.5 ])
# Estimate the density (named q, to emphesize that it does not need to be
# normalized). Parameter bw_method=0.48 is used to mimic the outcome of the
# kernelp function in Matlab.
q = stats.gaussian_kde(r, bw_method=0.48).evaluate(x)

# rejection sampling example
g_mean = 0
g_std = 1.1
g = stats.norm.pdf(x, loc=g_mean, scale=g_std)
# M is computed by discrete approximation
M = np.max(q/g)
# prescale
g *= M

# plot the densities
plt.figure()
plt.plot(x, q)
plt.plot(x, g, linestyle='--')
plt.fill_between(x, q)
plt.legend((r'$q(\theta|y)$', r'$Mg(\theta)$'))
plt.yticks(())
plt.title('Rejection sampling')
plt.ylim([0, 1.1*g.max()])

# illustrate one sample
r1 = -0.8
zi = np.argmin(np.abs(x-r1)) # find the closest grid point
plt.plot((x[zi], x[zi]), (0, q[zi]), color='gray')
plt.plot((x[zi], x[zi]), (q[zi], g[zi]), color='gray', linestyle='--')
r21 = 0.3 * g[zi]
r22 = 0.8 * g[zi]
plt.plot(r1, r21, marker='o', color='#4daf4a', markersize=12)
plt.plot(r1, r22, marker='o', color='#e41a1c', markersize=12)
# add annotations
plt.text(x[zi], q[zi], r'$\leftarrow \, q(\theta=r|y)$', fontsize=18)
plt.text(x[zi], g[zi], r'$\leftarrow \, g(\theta=r)$', fontsize=18)
plt.text(r1-0.1, r21, 'accepted', horizontalalignment='right')
plt.text(r1-0.1, r22, 'rejected', horizontalalignment='right')

# get nsamp samples
nsamp = 200
r1 = stats.norm.rvs(size=nsamp, loc=g_mean, scale=g_std)
zi = np.argmin(np.abs(x[:,None] - r1), axis=0)
r2 = np.random.rand(nsamp) * g[zi]
acc = r2 < q[zi]

# plot the densities againg
plotgrid = mpl.gridspec.GridSpec(2, 1, height_ratios=[5,1])
fig = plt.figure()
ax0 = plt.subplot(plotgrid[0])
plt.plot(x, q)
plt.plot(x, g, linestyle='--')
plt.fill_between(x, q)
plt.xticks(())
plt.yticks(())
plt.title('Rejection sampling')
plt.ylim([0, 1.1*g.max()])
plt.xlim((x[0],x[-1]))
# the samples
plt.scatter(r1[~acc], r2[~acc], 40, color='#ff999a')
plt.scatter(r1[acc], r2[acc], 40, color='#4daf4a')
plt.legend((r'$q(\theta|y)$', r'$Mg(\theta)$', 'rejected', 'accepted'))
# only accepted samples 
ax1 = plt.subplot(plotgrid[1])
plt.scatter(r1[acc], np.ones(np.count_nonzero(acc)), 40, color='#4daf4a', alpha=0.3)
plt.yticks(())
plt.xlim((x[0],x[-1]))
# add inter-axis lines
transf = fig.transFigure.inverted()
for i in range(nsamp):
    if acc[i] and x[0] < r1[i] and r1[i] < x[-1]:
        coord1 = transf.transform(ax0.transData.transform([r1[i], r2[i]]))
        coord2 = transf.transform(ax1.transData.transform([r1[i], 1]))
        fig.lines.append(mpl.lines.Line2D(
            (coord1[0], coord2[0]),
            (coord1[1], coord2[1]),
            transform=fig.transFigure,
            alpha=0.2
        ))

# alternative proposal distribution
g = np.empty(x.shape)
g[x <= -1.5] = np.linspace(q[0], np.max(q[x<=-1.5]), len(x[x<=-1.5]))
g[(x > -1.5) & (x <= 0.2)] = np.linspace(
    np.max(q[x<=-1.5]),
    np.max(q[(x>-1.5) & (x<=0.2)]),
    len(x[(x>-1.5) & (x<=0.2)])
)
g[(x > 0.2) & (x <= 2.3)] = np.linspace(
    np.max(q[(x>-1.5) & (x<=0.2)]),
    np.max(q[x>2.3]),
    len(x[(x>0.2) & (x<=2.3)])
)
g[x > 2.3] = np.linspace(np.max(q[x>2.3]), q[-1], len(x[x>2.3]))
M = np.max(q/g)
g *= M
# plot
plt.figure()
plt.plot(x, q)
plt.plot(x, g, linestyle='--')
plt.fill_between(x, q)
plt.legend((r'$q(\theta|y)$', r'$Mg(\theta)$'))
plt.yticks(())
plt.title('Rejection sampling - alternative proposal distribution')
plt.ylim([0, 1.1*g.max()])

plt.show()

