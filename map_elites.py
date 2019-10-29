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
# 
# from scipy.spatial import cKDTree : TODO
from sklearn.neighbors import KDTree, DistanceMetric
from sklearn.cluster import KMeans
from scipy.spatial import distance
import math
import numpy as np
import multiprocessing
from pathlib import Path
import kinematic_arm
import sys
import random
from collections import defaultdict
import GPy

global same_count
same_count = 0

global zero_count
zero_count = 0

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
        "dump_period": 10000,
        # do we use several cores?
        "parallel": True,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": True,
        # min/max of parameters
        "min": [0]*15,
        "max": [1]*15,
        "multi_task": False,
        "multi_mode": 'full'
    }
class Species:
    def __init__(self, x, desc, fitness):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = None
        self.challenges = 0

# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient

def scale(x,params):
    x_scaled = []
    for i in range(0,len(x)) :
        x_scaled.append(x[i] * (params["max"][i] - params["min"][i]) + params["min"][i])
    return np.array(x_scaled)

def variation(x, z, archive, params):
    y = x.copy()
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
                     n_init=1, n_jobs=-1, verbose=1)#,algorithm="full")
    k_means.fit(x)
    return k_means.cluster_centers_


def __make_hashable(array):
    return tuple(map(float, array))


# format: fitness, centroid, desc, genome \n
# fitness, centroid, desc and x are vectors
def __save_archive(archive, gen):
    def write_array(a, f):
        for i in a:
            f.write(str(i) + ' ')
    filename = 'archive_' + str(gen) + '.dat'
    with open(filename, 'w') as f:
        for k in archive.values():
            f.write(str(k.fitness) + ' ')
            write_array(k.centroid, f)
            write_array(k.desc, f)
            write_array(k.x, f)
            f.write("\n")


def __add_to_archive(s, archive, kdt):
    global same_count
    global zero_count
    niche_index = kdt.query([s.desc], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = __make_hashable(niche)
    s.centroid = n
    if(np.all(s.desc==0)):
        zero_count = zero_count + 1
    if n in archive:
        same_count= same_count + 1
        c = s.challenges + 1
        if s.fitness > archive[n].fitness:
            archive[n] = s
            return 1
        return 0
    else:
        archive[n] = s
        return 1


# evaluate a single vector (z) with a function f and return a species
# t = vector, function
def evaluate(t):
    # x (position) only useful in multi-task
    z, f, x, params = t 
    if params['multi_task']:
        fit, desc = f(z, x)
        return Species(z, x, fit)
    else:
        fit, desc = f(z)
        return Species(z, desc, fit)

# bandit opt
# probability matching / Adaptive pursuit Thierens GECCO 2005
# UCB: schoenauer / Sebag
def opt_tsize(successes, n_niches):
    n = 0
    for v in successes.values():
        n += len(v)
    v = [1, 10, 50, 100, 500, 1000]
    if len(successes.keys()) < len(v):
        return random.choice(v)
    ucb = []
    for k in v:
        x = [i[0] for i in successes[k]]
        mean = sum(x) / float(len(x)) * 100
        n_a = len(x)
        ucb += [mean +  math.sqrt(2 * math.log(n) / n_a)]
    a = np.argmax(ucb)
    t_size = v[a]
    return t_size


# map-elites algorithm (CVT variant)
def compute(dim_map=-1, dim_x=-1, f=None, n_niches=1000, num_evals=1e5, 
            centroids='cvt',
            params=default_params):
    print(params)
    assert(f != None)
    assert(dim_map != -1)
    assert(dim_x != -1)
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # create the centroids if needed
    c = None
    if centroids == 'cvt':
        c = __cvt(n_niches, dim_map,
                params['cvt_samples'], 
                params['cvt_use_cache'])
    elif type(centroids) is np.ndarray: # we can provide a 2D array, each row = 1 centroid
        c = centroids
    else:
        print("[map-elites] ERROR: unsupported centroid type => ", centroids)

    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    __write_centroids(c)

    # init archive (empty)
    archive = {}

    init_count = 0
    # main loop
    evals = 0
    b_evals = 0
    t_size = 1
    successes = defaultdict(list)
    while (evals < num_evals):
        to_evaluate = []
        if evals == 0:  # random initialization
            while(init_count<=params['random_init'] * n_niches):
                for i in range(0, params['random_init_batch']):
                    x = np.random.random(dim_x)
                    x = scale(x, params)
                    x_bounded = []
                    for i in range(0,len(x)):
                        elem_bounded = min(x[i],params["max"][i])
                        elem_bounded = max(elem_bounded,params["min"][i])
                        x_bounded.append(elem_bounded)
                    # random niche for multitask only (ignored otherwise)
                    rand_niche = np.random.random(dim_map)
                    # put in the pool to be evaluated
                    to_evaluate += [(np.array(x_bounded), f, rand_niche, params)]
                if params['parallel'] == True:
                    s_list = pool.map(evaluate, to_evaluate)
                else:
                    s_list = map(evaluate, to_evaluate)
                evals += len(to_evaluate)
                b_evals += len(to_evaluate)
                for s in s_list:
                    __add_to_archive(s, archive, kdt)
                init_count = len(archive)
                to_evaluate = []
        else:  # variation/selection loop
            keys = list(archive.keys())
            for n in range(0, params['batch_size']):
                # parent selection
                x = archive[keys[np.random.randint(len(keys))]]
                y = archive[keys[np.random.randint(len(keys))]]
                # copy & add variation
                z = variation(x.x, y.x, archive, params)
                if not params['multi_task']:
                    to_evaluate += [(z, f, x, params)]
                else:
                    # to_evaluate += multi_task_eval(c, archive, params)
                    # now decide where to evaluate
                    # randomly on the map (e.g. 10% niches)?
                    # in the neighborhood (fixed) with 10% niches?
                    # in the neighborhood (adaptive?)
                    # according to the distance between the parents (in behavioral space?)?
                    # in the neighborhood while expanding providing that it works?
                    # evaluate everywhere
                    if params['multi_mode'] == 'full':
                        for n in c: # for each centroid
                            to_evaluate += [(z, f, n, params)]
                    elif params['multi_mode'] == 'parents':
                        to_evaluate += [(z, f, x.desc, params)]
                        to_evaluate += [(z, f, y.desc, params)]
                    elif params['multi_mode'] == 'random':
                        niche = np.random.random(dim_map)
                        to_evaluate += [(z, f, niche, params)]
                    elif params['multi_mode'] == 'neighbors':
                        for p in [x, y]:
                            _, ind = kdt.query([p.desc], k=1)
                            for ii in ind[0]:
                                n = np.array(kdt.data[ii])
                                to_evaluate += [(z, f, n, params)]
                    elif params['multi_mode'] == 'neighbors_tournament':
                        niches = []
                        for p in range(0, params['n_size']):
                            niches += [np.random.random(dim_map)]
                        mn = min(niches, key=lambda xx: np.linalg.norm(xx - x.desc))
                        to_evaluate += [(z, f, mn, params)]
                    elif params['multi_mode'] == 'tournament_random':
                        t_size = np.random.randint(1, n_niches)
                        niches = []
                        for p in range(0, t_size):
                            niches += [np.random.random(dim_map)]
                        mn = min(niches, key=lambda xx: np.linalg.norm(xx - x.desc))
                        to_evaluate += [(z, f, mn, params)]
                    elif params['multi_mode'] == 'tournament_gp':
                        niches = []
                        for p in range(0, t_size):
                            niches += [np.random.random(dim_map)]
                        cd = distance.cdist(niches, [x.desc], 'euclidean')
                        mn = niches[np.argmin(cd)]
                        #mn = min(niches, key=lambda xx: np.linalg.norm(xx - x.desc))
                        to_evaluate += [(z, f, mn, params)]
                        # pareto sort density / challenges vs distance?
            # parallel evaluation of the fitness
            if params['parallel'] == True:
                s_list = pool.map(evaluate, to_evaluate)
            else:
                s_list = map(evaluate, to_evaluate)
            evals += len(to_evaluate)
            b_evals += len(to_evaluate)
            # natural selection
            suc = 0
            for s in s_list:
                suc += __add_to_archive(s, archive, kdt)
            if params['multi_mode'] == 'tournament_random' or params['multi_mode'] == 'tournament_gp':
                successes[t_size] += [(suc / params["batch_size"], evals)]
        if params['multi_mode'] == 'tournament_gp':# and (random.uniform(0, 1) < 0.05 or len(successes.keys()) < 5):
            t_size = opt_tsize(successes, n_niches)
        # write archive
        if params['dump_period'] != -1 and b_evals > params['dump_period']:
            __save_archive(archive, evals)
            b_evals = 0
            n_e = []
            for v in successes.values():
                n_e += [len(v)]
            print(evals, n_e)
            np.savetxt('t_size.dat', np.array(n_e))
    __save_archive(archive, evals)
    return archive


# a small test
if __name__ == "__main__":
    def rastrigin(xx):
        x = xx * 10.0 - 5.0
        f = 10 * x.shape[0]
        for i in range(0, x.shape[0]):
            f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
        return -f, np.array([xx[0], xx[1]])
    my_map = compute(dim_map=2, dim_x = 10, n_niches=1500, f=rastrigin)
