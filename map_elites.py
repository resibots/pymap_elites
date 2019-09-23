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
    assert(len(params["min"]) >= x.shape[0])
    assert(len(params["max"]) >= x.shape[0])
    y = x.copy()
    keys = list(archive.keys())
    z = archive[keys[np.random.randint(len(keys))]].x
    # this could be nicely vectorized
    for i in range(0,len(y)):
        # iso mutation
        a = np.random.normal(0, (params["max"][i]-params["min"][i])/300.0, 1)
        y[i] =  y[i] + a
        # line mutation
        b = np.random.normal(0, 20*(params["max"][i]-params["min"][i])/300.0, 1)
        y[i] =  y[i] + b*(x[i] - z[i])
    y_bounded = np.clip(y, a_min=params["min"][0:len(y)], a_max=params["max"][0:len(y)])
    return y_bounded

def uniform_random(dim_x, params):
    x = np.random.random(dim_x)
    return scale(x, params)


def gen_to_phen_direct(gen):
    return gen

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
        "min": [-1]*10,
        "max": [1]*10,
        # variation operator
        "variation" : variation,
        # operator for creating a random individual
        "random": uniform_random,
        # operator to transform a genotype to a phenotype (development)
        "gen_to_phen": gen_to_phen_direct,
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

# try to add s to the archive
# KDT is the kd-tree with the centroids (center of niches)
def __add_to_archive(s, archive, kdt):
    niche_index = kdt.query([s.desc], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = __make_hashable(niche)
    if n in archive:
        if s.fitness > archive[n].fitness:
            archive[n] = s
            return 1
    else:
        archive[n] = s
        return 1
    return 0

# evaluate a single vector (x) with a function f and return a species
# t = vector, function
def evaluate(t):
    x, z, f = t  # evaluate z (developped) with function f
    fit, desc = f(z)
    return Species(x, desc, fit)

# evaluate a list of potential solutions
# develop them to genotype first if needed
def eval_all(pool, to_evaluate, f, params):
    # apply dev and add the fitness to form a (x, f) tuple
    # we need to convert to list in python3
    to_evaluate_dev = list(zip(to_evaluate, map(params["gen_to_phen"], to_evaluate), [f]*len(to_evaluate)))
    if params['parallel'] == True:
        s_list = pool.map(evaluate, to_evaluate_dev)
    else:
        s_list = map(evaluate, to_evaluate_dev)
    return s_list

# map-elites algorithm (CVT variant)
def compute(dim_map, dim_x, f, n_niches=1000, n_gen=1000, params=default_params, archive={}, centroids=np.empty(shape=(0,0)), gen=0, pool=None):
    if pool==None:
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cores)
    # create the CVT
    if centroids.shape[0] == 0:
        centroids = __cvt(n_niches, dim_map,
                  params['cvt_samples'], params['cvt_use_cache'])
        __write_centroids(centroids)
    kdt = KDTree(centroids, leaf_size=30, metric='euclidean')

    init_count = 0
    successes = 0 # number of additions to archive (for external parameter control)
    # main loop
    for g in range(gen, gen + n_gen):
        to_evaluate = []
        if len(archive) == 0:  # random initialization
            print('init: ', end='', flush=True)
            while(init_count<=params['random_init'] * n_niches):
                for i in range(0, params['random_init_batch']):
                    x = params['random'](dim_x, params)
                    x_bounded = []
                    for i in range(0,len(x)):
                        elem_bounded = min(x[i],params["max"][i])
                        elem_bounded = max(elem_bounded,params["min"][i])
                        x_bounded.append(elem_bounded)
                    to_evaluate += [np.array(x_bounded)]
                s_list = eval_all(pool, to_evaluate, f, params)
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
                to_evaluate += [z]
            s_list = eval_all(pool, to_evaluate, f, params)
            print(str(len(s_list)) + ' ', end='', flush=True)
            # natural selection
            for s in s_list:
                successes += __add_to_archive(s, archive, kdt)
        # write archive
        if g % params['dump_period'] == 0 and params['dump_period'] != -1:
            fit_list = np.array([x.fitness for x in archive.values()])
            print("generation:{} size:{} max={} mean={}".format(g, len(archive.keys()), fit_list.max(), fit_list.mean()))
            __save_archive(archive, g, params['save_format'])
    __save_archive(archive, n_gen, params['save_format'])
    #pool.close()
    #del pool
    return archive, centroids, successes




# a small test
if __name__ == "__main__":
    def rastrigin(xx):
        x = xx * 10.0 - 5.0
        f = 10 * x.shape[0]
        for i in range(0, x.shape[0]):
            f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
        return -f, np.array([xx[0], xx[1]])

    archive, centroids = compute(2, 6, rastrigin, n_niches=5000, n_gen=2500)
