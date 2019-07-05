import glob
from pylab import *
import brewer2mpl
import numpy as np
import sys
import math

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

def load_file(fname, dim):
    print("loading ", fname, " dim=", dim)
    d = np.loadtxt(fname)
    data = {}
    # format: [fitness, centroid, desc, x]
    for i in range(0, d.shape[0]):
        n = __make_hashable(d[i][1:dim+1])
        data[n] = d[i][0]
    return data


def load(dir, dim):
    f_list = glob.glob(dir + '/archive_*.dat')
    evals = []
    data = []
    for i in f_list:
        e = int(i.split('_')[1].split('.')[0])
        evals += [e]
    evals = sorted(evals)
    for i in evals:
        data += [load_file(dir + "/archive_" + str(i) + ".dat", dim)]
    return data, evals
    

def perc(data):
    median = np.zeros(data.shape[1])
    perc_25 = np.zeros(data.shape[1])
    perc_75 = np.zeros(data.shape[1])
    for i in range(0, len(median)):
        median[i] = np.median(data[:, i])
        perc_25[i] = np.percentile(data[:, i], 25)
        perc_75[i] = np.percentile(data[:, i], 75)
    return median, perc

def compute_precision(a, ref):
    keys = list(a.keys())
    diff = 0
    for k in keys:
        diff += a[k]
    return diff / len(keys)

def compute_coverage(a, ref):
    keys_ref = list(ref.keys())
    keys_a = list(a.keys())
    c = 0.0
    for k in keys_ref:
        if k in keys_a:
            c += 1
    return c / len(keys_ref)



def compute_diff(dim, ref_fname, data_dir):
    ref = load_file(ref_fname, dim)
    data, evals = load(data_dir, dim)
    precision = []
    coverage = []
    for i in range(0, len(data)):
        p = compute_precision(data[i], ref)
        precision += [p]
        coverage += [compute_coverage(data[i], ref)]
    print(precision[-1])
    return precision, coverage, evals


fig = figure() # no frame
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

k = 0
dim = 2
ref_fname = sys.argv[1]
for i in sys.argv[2:]:
    p, c, e = compute_diff(dim, ref_fname, i)
    ax1.plot(e, p, linewidth=1, color=colors[k%len(colors)], label=i)
    ax2.plot(e, c, linewidth=1, color=colors[k%len(colors)], label=i)
    k += 1
# now all plot function should be applied to ax
#ax.fill_between(x, perc_25_low_mut, perc_75_low_mut, alpha=0.25, linewidth=0, color=colors[0]) 
#ax.fill_between(x, perc_25_high_mut, perc_75_high_mut, alpha=0.25, linewidth=0, color=colors[1])
#ax.plot(x, med_low_mut, linewidth=2, color=colors[0])
#ax.plot(x, med_high_mut, linewidth=2, linestyle='--', color=colors[1])

# change xlim to set_xlim
ax1.set_xlim(0, 50000)
ax2.set_xlim(0, 50000)

#ax.set_ylim(-5000, 300)

#change xticks to set_xticks
#ax.set_xticks(np.arange(0, 500, 100))

legend = ax2.legend(loc=4)#bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=(3))
#frame = legend.get_frame()
#frame.set_facecolor('1.0')
#frame.set_edgecolor('1.0')

fig.savefig('progress.pdf')