import glob
from pylab import *
import brewer2mpl
import numpy as np
import sys
import math
import gzip
import matplotlib.gridspec as gridspec
from scipy import stats

from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def customize_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.minorticks_on()
    ax.grid(which='minor', linestyle='-', linewidth='0.5', alpha=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    # offset the spines
    for spine in ax.spines.values():
     spine.set_position(('outward', 5))

    # put the grid behind
    ax.set_axisbelow(True)
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)


fig = figure(frameon=False,figsize=(6, 4))
ax1 = fig.add_subplot(111)

ax1.set_title('Cumulative density (unnormalized)')
my_cmap = cm.viridis
norm = matplotlib.colors.Normalize(vmin=0, vmax=len(sys.argv))

def cdf(x, data):
    res = []
    med = None
    for v in x:
        s = sum(v < data)
        res += [s]
        if med == None and s < (len(data) / 2):
            print(len(data), s)
            med = (v, s)
    return res, med

for i in range(1, len(sys.argv)):
    data = np.loadtxt(sys.argv[i])[:,0]
    # unbinned CDF
    # https://stackoverflow.com/questions/10640759/how-to-get-the-cumulative-distribution-function-with-numpy#comment52345953_32230314
    x = np.sort(data)[::-1]
    y = np.array(range(len(data)))
    median = (np.median(data), np.median(y))
    ax1.semilogy(x, y, lw=2, label=sys.argv[i], color=cm.viridis(norm(i)))
    ax1.semilogy([median[0], median[0]], [median[1],0], '--', color=cm.viridis(norm(i)), lw=0.75)
    ax1.semilogy([median[0], median[0]], [median[1],0], 'o', color=cm.viridis(norm(i)), markersize=8, mfc='white',markeredgewidth=1)


customize_axis(ax1)

legend = ax1.legend()#bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=(3))
frame = legend.get_frame()
frame.set_facecolor('0.9')
frame.set_edgecolor('1.0')

fig.tight_layout()
fig.savefig('density.pdf')
fig.savefig('density.svg')
