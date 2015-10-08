"""Bayesian data analysis
Chapter 11, demo 1

Gibbs sampling demonstration

"""

from __future__ import division
import threading
import numpy as np
import scipy.io # For importing a matlab file
from scipy import linalg, stats
import matplotlib as mpl
import matplotlib.pyplot as plt

# edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2, markeredgewidth=1.5,
       markersize=8)
plt.rc('axes', color_cycle=('#377eb8','#e41a1c','#4daf4a',
                            '#984ea3','#ff7f00','#ffff33'))

# Parameters of a Normal distribution used as a toy target distribution
y1 = 0
y2 = 0
r = 0.8
S = np.array([[1.0, r], [r, 1.0]])

# Starting value of the chain
t1 = -2.5
t2 = 2.5
# Number of iterations.
M = 2*1000
# N.B. In this implementation one iteration updates only one parameter and one
# complete iteration updating both parameters takes two basic iterations. This
# implementation was used to make plotting of Gibbs sampler's zig-zagging. In
# plots You can implement this also by saving only the final state of complete
# iteration updating all parameters.

# ====== Gibbs sampling here

# Allocate memory for the samples
tt = np.empty((M,2))
tt[0] = [t1, t2]    # Save starting point

# For demonstration load pre-computed values
# Replace this with your algorithm!
# tt is a M x 2 array, with M samples of both theta_1 and theta_2
res_path = '../utilities_and_data/demo11_2.mat'
res = scipy.io.loadmat(res_path)
''' Content information of the precalculated results:
>>> scipy.io.whosmat(res_path)
[('tt', (2001, 2), 'double')]
'''
tt = res['tt']

# ====== The rest is just for illustration

# Grid
Y1 = np.linspace(-4.5, 4.5, 150)
Y2 = np.linspace(-4.5, 4.5, 150)

# Plot 90% HPD.
# In 2d-case contour for 90% HPD is an ellipse, whose semimajor
# axes can be computed from the eigenvalues of the covariance
# matrix scaled by a value selected to get ellipse match the
# density at the edge of 90% HPD. Angle of the ellipse could be 
# computed from the eigenvectors, but since marginals are same
# we know that angle is 45 degrees.
q = np.sort(np.sqrt(linalg.eigh(S, eigvals_only=True)) * 2.147)
el = mpl.patches.Ellipse(
    xy = (y1,y2),
    width = 2 * q[1],
    height = 2 * q[0],
    angle = 45,
    facecolor = 'none',
    edgecolor = '#e41a1c'
)
el_legend = mpl.lines.Line2D([], [], color='#e41a1c', linewidth=1)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, aspect='equal')
ax.add_artist(el)
samp_legend, = ax.plot(
    tt[0,0], tt[0,1], 'o', markerfacecolor='none', markeredgecolor='#377eb8')
ax.set_xlim([-4.5, 4.5])
ax.set_ylim([-4.5, 4.5])
ax.set_xlabel(r'$\theta_1$', fontsize=18)
ax.set_ylabel(r'$\theta_2$', fontsize=18)
htext = ax.set_title('Gibbs sampling\npress any key to continue...',
                     fontsize=18)
ax.legend((el_legend, samp_legend), ('90% HPD', 'Starting point'), numpoints=1,
          loc='lower right')
pdfline_legend = mpl.lines.Line2D([], [], color='#377eb8')
chain_legend = mpl.lines.Line2D(
    [], [], color='#377eb8', marker='o',
    markerfacecolor='none', markeredgecolor='#377eb8'
)
burnchain_legend = mpl.lines.Line2D(
    [], [], color='m', marker='o',
    markerfacecolor='none', markeredgecolor='m'
)

# function for interactively updating the figure
def update_figure(event):
    
    if icontainer.stage == 0 and icontainer.i < 7 and icontainer.drawdist:
        i = icontainer.i
        icontainer.drawdist = False
        # Remove previous lines
        for l in icontainer.remove_lines:
            ax.lines.remove(l)
        icontainer.remove_lines = []
        if i % 2 == 0:
            line = ax.axhline(y=tt[i,1], linestyle='--', color='k')
            icontainer.remove_lines.append(line)
            line, = ax.plot(
                Y1,
                tt[i,1] + stats.norm.pdf(
                    Y1,
                    loc = y1 + r*(tt[i,1] - y2),
                    scale = np.sqrt((1 - r**2))
                ),
                color = '#377eb8'
            )
            icontainer.remove_lines.append(line)
            if i == 0:
                ax.legend(
                    (el_legend, samp_legend, pdfline_legend),
                    ( '90% HPD',
                      'Starting point',
                     r'Conditional density given $\theta_2$'
                    ),
                    numpoints=1,
                    loc='lower right'
                )
            else:
                ax.legend(
                    (el_legend, samp_legend, pdfline_legend),
                    ( '90% HPD',
                      'Samples from the chain',
                     r'Conditional density given $\theta_2$'
                    ),
                    loc='lower right'
                )
        else:
            line = ax.axvline(x=tt[i,0], linestyle='--', color='k')
            icontainer.remove_lines.append(line)
            line, = ax.plot(
                tt[i,0] + stats.norm.pdf(
                    Y2,
                    loc = y2 + r*(tt[i,0] - y1),
                    scale = np.sqrt((1 - r**2))
                ),
                Y2,
                color = '#377eb8'
            )
            icontainer.remove_lines.append(line)
            ax.legend(
                (el_legend, samp_legend, pdfline_legend),
                ( '90% HPD',
                  'Samples from the chain',
                 r'Conditional density given $\theta_1$'
                ),
                loc='lower right'
            )
        
        fig.canvas.draw()
    
    elif icontainer.stage == 0 and icontainer.i < 7 and not icontainer.drawdist:
        icontainer.i += 1
        i = icontainer.i
        if i == 6:
            icontainer.stage += 1
        icontainer.drawdist = True
        sampi, = ax.plot(tt[i,0], tt[i,1], 'o', markerfacecolor='none',
                markeredgecolor='#377eb8')
        icontainer.samps.append(sampi)
        if i == 1:
            ax.legend(
                (el_legend, samp_legend, pdfline_legend),
                ( '90% HPD',
                  'Samples from the chain',
                 r'Conditional density given $\theta_2$'
                ),
                loc='lower right'
            )
        fig.canvas.draw()
    
    elif icontainer.stage == 1:
        icontainer.stage += 1
        for l in icontainer.remove_lines:
            ax.lines.remove(l)
        icontainer.remove_lines = []
        ax.legend(
            (el_legend, samp_legend),
            ('90% HPD', 'Samples from the chain'),
            loc='lower right'
        )
        fig.canvas.draw()
    
    elif icontainer.stage == 2:
        icontainer.stage += 1
        for s in icontainer.samps:
            ax.lines.remove(s)
        icontainer.samps = []
        line, = ax.plot(
            tt[:icontainer.i+1,0], tt[:icontainer.i+1,1], color='#377eb8')
        icontainer.samps.append(line)
        line, = ax.plot(
            tt[:icontainer.i+1:2,0], tt[:icontainer.i+1:2,1],
            'o', markerfacecolor='none', markeredgecolor='#377eb8')
        icontainer.samps.append(line)
        ax.legend((el_legend, chain_legend), ('90% HPD', 'Markov chain'),
                  loc='lower right')
        fig.canvas.draw()
    
    elif icontainer.stage == 3:
        icontainer.stage += 1
        # modify helper text
        htext.set_text('Gibbs sampling\npress `q` to skip animation')
        # start the timer
        anim_thread.start()
    
    elif icontainer.stage == 4 and event.key == 'q':
        # stop the animation
        stop_anim.set()
        
    elif icontainer.stage == 5:
        icontainer.stage += 1
        for s in icontainer.samps:
            ax.lines.remove(s)
        icontainer.samps = []
        # remove helper text
        icontainer.itertext.remove()
        line, = ax.plot(tt[:burnin,0], tt[:burnin,1], color='m')
        icontainer.samps.append(line)
        line, = ax.plot(tt[:burnin:2,0], tt[:burnin:2,1], 'o',
                markerfacecolor='none', markeredgecolor='m')
        icontainer.samps.append(line)
        line, = ax.plot(
            tt[burnin:nanim+1,0], tt[burnin:nanim+1,1], color='#377eb8')
        icontainer.samps.append(line)
        line, = ax.plot(tt[burnin:nanim+1:2,0], tt[burnin:nanim+1:2,1], 'o',
                markerfacecolor='none', markeredgecolor='#377eb8')
        icontainer.samps.append(line)
        ax.legend(
            (el_legend, chain_legend, burnchain_legend),
            ('90% HPD', 'Markov chain', 'warm-up'),
            loc='lower right'
        )
        fig.canvas.draw()
        
    elif icontainer.stage == 6:
        icontainer.stage += 1
        for s in icontainer.samps:
            ax.lines.remove(s)
        icontainer.samps = []
        line, = ax.plot(tt[burnin:nanim+1:2,0], tt[burnin:nanim+1:2,1], 'o',
                markerfacecolor='none', markeredgecolor='#377eb8')
        icontainer.samps.append(line)
        ax.legend(
            (el_legend, samp_legend),
            ('90% HPD', 'samples from the chain after warm-up'),
            loc='lower right'
        )
        fig.canvas.draw()
        
    elif icontainer.stage == 7:
        icontainer.stage += 1
        for s in icontainer.samps:
            ax.lines.remove(s)
        icontainer.samps = []
        points = ax.scatter(
            tt[burnin::2,0], tt[burnin::2,1], 10, alpha=0.5, color='#377eb8')
        icontainer.samps.append(points)
        ax.legend(
            (el_legend, points),
            ('90% HPD', '950 samples from the chain'),
            loc='lower right'
        )
        fig.canvas.draw()
        
    elif icontainer.stage == 8:
        icontainer.stage += 1
        fig.clear()
        
        indexes = np.arange(burnin,M,2)
        samps = tt[indexes]
        
        ax1 = fig.add_subplot(3,1,1)
        ax1.axhline(y=0, linewidth=1, color='gray')
        line1, line2, = ax1.plot(indexes/2, samps, linewidth=1)
        ax1.legend((line1, line2), (r'$\theta_1$', r'$\theta_2$'))
        ax1.set_xlabel('iteration')
        ax1.set_title('trends')
        ax1.set_xlim([burnin/2, 1000])
        
        ax2 = fig.add_subplot(3,1,2)
        ax2.axhline(y=0, linewidth=1, color='gray')
        ax2.plot(
            indexes/2,
            np.cumsum(samps, axis=0)/np.arange(1,len(samps)+1)[:,None],
            linewidth=1.5
        )
        ax2.set_xlabel('iteration')
        ax2.set_title('cumulative average')
        ax2.set_xlim([burnin/2, 1000])
        
        ax3 = fig.add_subplot(3,1,3)
        maxlag = 20
        sampsc = samps - np.mean(samps, axis=0)
        acorlags = np.arange(maxlag+1)
        ax3.axhline(y=0, linewidth=1, color='gray')
        for i in [0,1]:
            t = np.correlate(sampsc[:,i], sampsc[:,i], 'full')
            t = t[-len(sampsc):-len(sampsc)+maxlag+1] / t[-len(sampsc)]
            ax3.plot(acorlags, t)
        ax3.set_xlabel('lag')
        ax3.set_title('estimate of the autocorrelation function')
        
        fig.suptitle('Gibbs sampling - press any key to continue...',
                     fontsize=18)
        fig.subplots_adjust(hspace=0.6)
        fig.canvas.draw()
        
    elif icontainer.stage == 9:
        icontainer.stage += 1
        fig.clear()
        
        indexes = np.arange(burnin,M,2)
        samps = tt[indexes]
        nsamps = np.arange(1,len(samps)+1)
        
        ax1 = fig.add_subplot(1,1,1)
        ax1.axhline(y=0, linewidth=1, color='gray')
        line1, line2, = ax1.plot(
            indexes/2,
            np.cumsum(samps, axis=0)/nsamps[:,None],
            linewidth=1.5
        )
        er1, = ax1.plot(
            indexes/2, 1.96/np.sqrt(nsamps/4), 'k--', linewidth=1)
        ax1.plot(indexes/2, -1.96/np.sqrt(nsamps/4), 'k--', linewidth=1)
        er2, = ax1.plot(
            indexes/2, 1.96/np.sqrt(nsamps), 'k:', linewidth=1)
        ax1.plot(indexes/2, -1.96/np.sqrt(nsamps), 'k:', linewidth=1)
        ax1.set_xlabel('iteration')
        ax1.set_title('Gibbs sampling\ncumulative average')
        ax1.legend(
            (line1, line2, er1, er2),
            (r'$\theta_1$', r'$\theta_2$',
              '95% interval for MCMC error',
              '95% interval for independent MC'
            )
        )
        ax1.set_xlim([burnin/2, 1000])
        ax1.set_ylim([-2, 2])
        fig.canvas.draw()
        
# function for performing the figure animation in thread
def animation():
    icontainer.itertext = ax.text(-4, 4, '', fontsize=18)
    delay0 = 0.4
    delayk = 0.85
    while icontainer.i < nanim:
        icontainer.i += 1
        i = icontainer.i
        icontainer.itertext.set_text('iter {}'.format(i//2))
        # show next sample
        line, = ax.plot(tt[i-1:i+1,0], tt[i-1:i+1,1], color='#377eb8')
        icontainer.samps.append(line)
        if i % 2 == 0:
            line, = ax.plot(
                tt[i,0], tt[i,1], 'o',
                markerfacecolor='none', markeredgecolor='#377eb8')
            icontainer.samps.append(line)
        # update figure
        fig.canvas.draw()
        if i < nanim and (i < 16 or i % 2 == 0):
            # wait animation delay time or until animation is cancelled
            stop_anim.wait(delay0)
            delay0 *= delayk
        if stop_anim.isSet():
            # animation cancelled
            break
    # skip the rest if the figure does not exist anymore
    if not plt.fignum_exists(fig.number):
        return    
    # advance stage
    icontainer.stage += 1
    # modify helper text
    htext.set_text('Gibbs sampling\npress any key to continue...')
    # plot the rest of the samples
    if i < nanim:
        icontainer.itertext.set_text('iter {}'.format(nanim//2))
        line, = ax.plot(tt[i:nanim+1,0], tt[i:nanim+1,1], color='#377eb8')
        icontainer.samps.append(line)
        line, = ax.plot(tt[nanim:i-1:-2,0], tt[nanim:i-1:-2,1], 'o',
                markerfacecolor='none', markeredgecolor='#377eb8')
        icontainer.samps.append(line)
        icontainer.i = nanim
    fig.canvas.draw()

# animation related variables
stop_anim = threading.Event()
anim_thread = threading.Thread(target=animation)
nanim = 200
burnin = 50

# store the information of the current stage of the figure
class icontainer(object):
    stage = 0
    i = 0
    drawdist = True
    remove_lines = []
    samps = [samp_legend]
    itertext = None

# set figure to react to keypress events
fig.canvas.mpl_connect('key_press_event', update_figure)
# start blocking figure
plt.show()

