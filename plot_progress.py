import glob
from pylab import *
import brewer2mpl
import numpy as np
import sys
import math
from collections import defaultdict

 # brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors
 
params = {
    'axes.labelsize': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [8.5, 8.5]
}
rcParams.update(params)



def __make_hashable(array):
    return tuple(map(float, array))

def get_evals(dir):
    d_list = glob.glob(dir + '/*')
    f_list = glob.glob(d_list[0] + '/archive_*.dat')
    evals = []
    for i in f_list:
        e = int(i.split('_')[-1].split('.')[0])
        evals += [e]
    evals = sorted(evals)
    return evals

def load_file(fname, dim):
    print("loading ", fname, " dim=", dim)
    d = np.loadtxt(fname)
    data = {}
    # format: [fitness, centroid, desc, x]
    for i in range(0, d.shape[0]):
        n = __make_hashable(d[i][1:dim+1])
        data[n] = d[i][0]
    return data

# return a dictionnary of lists with the average fitness
def load_treatment(dir, dim):
    evals = get_evals(dir)
    d_list = glob.glob(dir + '/*')
    data = {}
    for i in evals:
        data[i] = []
        for d in d_list:
            x = load_file(d +  "/archive_" + str(i) + ".dat", dim)
            keys = list(x.keys())
            f = 0
            for j in keys:
                f += x[j]
            f /= len(x)
            data[i] += [f]
    return data
    
# process all the replicates of a single treatment
def process_treatment(data):
    keys = list(data)
    median = np.zeros(len(keys))
    perc_25 = np.zeros(len(keys))
    perc_75 = np.zeros(len(keys))
    i = 0
    for k in keys:
        median[i] = np.median(data[k])
        perc_25[i] = np.percentile(data[k], 25)
        perc_75[i] = np.percentile(data[k], 75)
        i += 1
    return median, perc_25, perc_75


fig = figure() # no frame
ax1 = fig.add_subplot(111)

k = 0
dim = 2
for i in sys.argv[2:]:
    data = load_treatment(i, dim)
    m, p25, p75 = process_treatment(data)
    x = list(data.keys())
    ax1.fill_between(x, p25, p75, alpha=0.25, linewidth=0, color=colors[k%len(colors)]) 
    ax1.plot(x, m, linewidth=1, color=colors[k%len(colors)], label=i)
    k += 1
# now all plot function should be applied to ax
#ax.fill_between(x, perc_25_low_mut, perc_75_low_mut, alpha=0.25, linewidth=0, color=colors[0]) 
#ax.fill_between(x, perc_25_high_mut, perc_75_high_mut, alpha=0.25, linewidth=0, color=colors[1])
#ax.plot(x, med_low_mut, linewidth=2, color=colors[0])
#ax.plot(x, med_high_mut, linewidth=2, linestyle='--', color=colors[1])

# change xlim to set_xlim
#ax1.set_xlim(0, 50000)
#ax2.set_xlim(0, 50000)

#ax.set_ylim(-5000, 300)

#change xticks to set_xticks
#ax.set_xticks(np.arange(0, 500, 100))

legend = ax1.legend(loc=4)#bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=(3))
#frame = legend.get_frame()
#frame.set_facecolor('1.0')
#frame.set_edgecolor('1.0')

fig.savefig('progress_dim' + str(dim) + '.pdf')