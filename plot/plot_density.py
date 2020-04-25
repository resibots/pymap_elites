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

def customize_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.tick_params(axis='y', length=0)
    #ax.get_yaxis().tick_left()

    # offset the spines
    for spine in ax.spines.values():
     spine.set_position(('outward', 5))
    # put the grid behind
    ax.set_axisbelow(True)
    ax.grid(axis='y', color="0.9", linestyle='--', linewidth=1)


fig = figure(frameon=False,figsize=(10, 7))
ax1 = fig.add_subplot(111)

ax1.set_title('Density')
my_cmap = cm.viridis

norm = matplotlib.colors.Normalize(vmin=0, vmax=len(sys.argv))

for i in range(1, len(sys.argv)):
    data = np.loadtxt(sys.argv[i])[:,0]
    density = stats.kde.gaussian_kde(data)
    #density.covariance_factor = lambda : .1
    #density._compute_covariance()
    print('density computed')
    x = np.linspace(start=min(data), stop=max(data), num=500)
    y = density(x)
    print('plotting...')
    ax1.plot(x, density(x), lw=2, label=sys.argv[i], color=cm.viridis(norm(i)))

customize_axis(ax1)

legend = ax1.legend()#bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=(3))
frame = legend.get_frame()
frame.set_facecolor('0.9')
frame.set_edgecolor('1.0')

fig.tight_layout()
fig.savefig('density.pdf')
fig.savefig('density.svg')
