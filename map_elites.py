#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
import math
import numpy as np
import multiprocessing
from pathlib import Path

global same_count
same_count = 0

global zero_count
zero_count = 0

class Species:
    def __init__(self, x, desc, fitness):
        self.x = x
        self.desc = desc
        self.fitness = fitness

def scale(x,params):
    x_scaled = []
    for i in range(0,len(x)) :
        x_scaled.append(x[i] * (params["max"][i] - params["min"][i]) + params["min"][i])
    return np.array(x_scaled)

def variation(x, archive, params):
    y = x.copy()
    keys = list(archive.keys())
    z = archive[keys[np.random.randint(len(keys))]].x
    for i in range(0,len(y)):
        # iso mutation
        a = np.random.normal(0, (params["max"][i]-params["min"][i])/300.0, 1)
        y[i] =  y[i] + a
        # line mutation
        b = np.random.normal(0, 20*(params["max"][i]-params["min"][i])/300.0, 1)
        y[i] =  y[i] + b*(x[i] - z[i])
    y_bounded = []
    for i in range(0,len(y)):
        elem_bounded = min(y[i],params["max"][i])
        elem_bounded = max(elem_bounded,params["min"][i])
        y_bounded.append(elem_bounded)
    return np.array(y_bounded)


default_params = \
    {
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to paralleliez
        "batch_size": 100,
        # proportion of niches to be filled before starting
        "random_init": 0.1,
        # batch for random initialization
        "random_init_batch": 100,
        # parameters of the "mutation" operator
        "sigma_iso": 0.01,
        # parameters of the "cross-over" operator
        "sigma_line": 0.2,
        # when to write results (one generation = one batch)
        "dump_period": 100,
        # do we use several cores?
        "parallel": True,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": True,
        # min/max of parameters
        "min": [0,0,-1,0,0,0],
        "max": [0.1,10,0,1,1,1,1],
        # variation operator
        "variation" : variation,
        # save in 'bin' or 'txt'
        "save_format":'txt'
    }

def __centroids_filename(k, dim):
    return 'centroids_' + str(k) + '_' + str(dim) + '.dat'


def __write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')


def __cvt(k, dim, samples, cvt_use_cache=True):
    # check if we have cached values
    if cvt_use_cache:
        fname = __centroids_filename(k, dim)
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    # otherwise, compute cvt
    x = np.random.rand(samples, dim)
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, n_jobs=-1, verbose=1,algorithm="full")
    k_means.fit(x)
    return k_means.cluster_centers_


def __make_hashable(array):
    return tuple(map(float, array))

def archive_to_array(archive):
    v = list(archive.values())[0]
    d_desc = v.desc.shape[0]
    d_vector = v.x.shape[0]
    n_solutions = len(archive.values())
    # fit, desc, x
    a = np.zeros((n_solutions, 1 + d_desc + d_vector))
    n = 0
    for k in archive.values():
        a[n, 0] = k.fitness
        a[n, 1:d_desc+1] = k.desc
        a[n, d_desc+1:a.shape[1]] = k.x
        n += 1
    return a
    
# format: centroid fitness desc x \n
# centroid, desc and x are vectors
def __save_archive(archive, gen, format='bin'):
    a = archive_to_array(archive)
    filename = 'archive_' + str(gen)
    if format == 'txt':
        np.savetxt(filename + '.dat', a)
    else:
        np.save(filename + '.npy', a)


def __add_to_archive(s, archive, kdt):
    global same_count
    global zero_count
    niche_index = kdt.query([s.desc], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = __make_hashable(niche)
    if(np.all(s.desc==0)):
        zero_count = zero_count + 1
    if n in archive:
        same_count= same_count + 1
        if s.fitness > archive[n].fitness:
            archive[n] = s
    else:
        archive[n] = s
# evaluate a single vector (x) with a function f and return a species
# t = vector, function
def evaluate(t):
    z, f = t  # evaluate z with function f
    fit, desc = f(z)
    return Species(z, desc, fit)

# map-elites algorithm (CVT variant)
def compute(dim_map, dim_x, f, n_niches=1000, n_gen=1000, params=default_params):
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # create the CVT
    c = __cvt(n_niches, dim_map,
              params['cvt_samples'], params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    __write_centroids(c)

    # init archive (empty)
    archive = {}

    init_count = 0
    # main loop
    for g in range(0, n_gen + 1):
        to_evaluate = []
        if g == 0:  # random initialization
            print('init: ', end='', flush=True)
            while(init_count<=params['random_init'] * n_niches):
                for i in range(0, params['random_init_batch']):
                    x = np.random.random(dim_x)
                    x = scale(x, params)
                    x_bounded = []
                    for i in range(0,len(x)):
                        elem_bounded = min(x[i],params["max"][i])
                        elem_bounded = max(elem_bounded,params["min"][i])
                        x_bounded.append(elem_bounded)
                    to_evaluate += [(np.array(x_bounded), f)]
                if params['parallel'] == True:
                    s_list = pool.map(evaluate, to_evaluate)
                else:
                    s_list = map(evaluate, to_evaluate)
                for s in s_list:
                    __add_to_archive(s, archive, kdt)
                init_count = len(archive)
                print("[{}/{}] ".format(init_count, int(params['random_init'] * n_niches)), end='', flush=True)
                to_evaluate = []
        else:  # variation/selection loop
            keys = list(archive.keys())
            for n in range(0, params['batch_size']):
                # parent selection
                x = archive[keys[np.random.randint(len(keys))]]
                # copy & add variation
                z = params["variation"](x.x, archive, params)
                to_evaluate += [(z, f)]
            # parallel evaluation of the fitness
            if params['parallel'] == True:
                s_list = pool.map(evaluate, to_evaluate)
            else:
                s_list = map(evaluate, to_evaluate)
            print(str(len(s_list)) + ' ', end='', flush=True)
            # natural selection
            for s in s_list:
                __add_to_archive(s, archive, kdt)
        # write archive
        if g % params['dump_period'] == 0 and params['dump_period'] != -1:
            print("generation:", g, " archive size:", len(archive.keys()))
            __save_archive(archive, g, params['save_format'])
    __save_archive(archive, n_gen, params['save_format'])
    return archive




# a small test
if __name__ == "__main__":
    def rastrigin(xx):
        x = xx * 10.0 - 5.0
        f = 10 * x.shape[0]
        for i in range(0, x.shape[0]):
            f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
        return -f, np.array([xx[0], xx[1]])

    archive = compute(2, 6, rastrigin, n_niches=5000, n_gen=2500)
