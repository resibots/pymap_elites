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

import math
import numpy as np
import multiprocessing
from pathlib import Path
import sys
import random
from collections import defaultdict
from sklearn.neighbors import KDTree
from scipy.spatial import distance

from map_elites import common as cm


# TODO : we do not need the KD-tree here -> use the archive directly
# TODO : remove the to_evaluate_centroid


# evaluate a single vector (z) with a function f and return a species
# t = vector, function
def evaluate(t):
    z, f, x, params = t 
    fit, desc = f(z, x)
    return cm.Species(z, x, fit)

# bandit opt for optimizing tournament size
# probability matching / Adaptive pursuit Thierens GECCO 2005
# UCB: schoenauer / Sebag
def opt_tsize(successes, n_niches):
    n = 0
    for v in successes.values():
        n += len(v)
    v = [1, 10, 50, 100, 500]#, 1000]
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

# select the niche according to
def select_niche(x, z, f, c, tasks, t_size, params):
    to_evaluate = []
    to_evaluate_centroid = []
    if params['multi_mode'] == 'full':
        # evaluate on all the niches
        for i in range(0, c.shape[0]): # for each centroid
            to_evaluate += [(z, f, tasks[i], params)]
            to_evaluate_centroid += [c[i, :]]
    elif params['multi_mode'] == 'parents':
        # evaluate on the niche of the parents
        to_evaluate += [(z, f, x.desc, params)]
        to_evaluate += [(z, f, y.desc, params)]
    elif params['multi_mode'] == 'random':
        # evaluate on a random niche
        niche = np.random.randint(c.shape[0])
        to_evaluate += [(z, f, tasks[niche], params)]
        to_evaluate_centroid += [c[niche, :]]
    elif params['multi_mode'] == 'neighbors':
        # evaluate on the nearest neighbor of each parent
        for p in [x, y]:
            _, ind = kdt.query([p.desc], k=1)
            for ii in ind[0]:
                n = np.array(kdt.data[ii])
                to_evaluate += [(z, f, n, params)]
    elif params['multi_mode'] == 'neighbors_tournament':
        # do a tournmanent to find the niche that is close to x
        # (parametrized by n_size)
        niches = []
        for p in range(0, params['n_size']):
            niches += [np.random.random(dim_map)]
        mn = min(niches, key=lambda xx: np.linalg.norm(xx - x.desc))
        to_evaluate += [(z, f, mn, params)]
    elif params['multi_mode'] == 'tournament_random':
        # do a tournmanent to find the niche that is close to x
        # (with a random size)
        t_size = np.random.randint(1, n_niches)
        niches = []
        for p in range(0, t_size):
            niches += [np.random.random(dim_map)]
        mn = min(niches, key=lambda xx: np.linalg.norm(xx - x.desc))
        to_evaluate += [(z, f, mn, params)]
    elif params['multi_mode'] == 'bandit_niche' or params['multi_mode'] == 'tournament':
        #print("tsize:", t_size)
        # we select the parent (a single one), then we select the niche
        # tournament using the bandit
        niches_centroids = []
        niches_tasks = []
        for p in range(0, t_size):
            n = np.random.randint(c.shape[0])
            niches_centroids += [c[n, :]]
            niches_tasks += [tasks[n]]
        #print('xcentroid:', x.centroid)
        cd = distance.cdist(niches_centroids, [x.centroid], 'euclidean')
        cd_min = np.argmin(cd)
        to_evaluate += [(z, f, niches_tasks[cd_min], params)]
        to_evaluate_centroid += [niches_centroids[cd_min]]
    return to_evaluate, to_evaluate_centroid

# map-elites algorithm (CVT variant)
def compute(dim_map=-1, dim_x=-1, f=None, n_niches=1000, num_evals=1e5, 
            centroids='cvt',
            tasks=[], 
            params=cm.default_params,
            log_file=None):
    print(params)
    assert(f != None)
    assert(dim_map != -1)
    assert(dim_x != -1)
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # create the centroids if needed
    c = None
    if type(centroids) is str and centroids == 'cvt':
        print('using CVT for centroids')
        c = cm.__cvt(n_niches, dim_map,
                params['cvt_samples'], 
                params['cvt_use_cache'])
    elif type(centroids) is np.ndarray: # we can provide a 2D array, each row = 1 centroid
        c = centroids
        n_niches = centroids.shape[0]
    else:
        print("[map-elites] ERROR: unsupported centroid type => ", centroids)

    if tasks == []:
        tasks = c

    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    cm.__write_centroids(c)
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
        to_evaluate_centroid = []
        if evals == 0 or init_count<=params['random_init'] * n_niches:
            for i in range(0, params['random_init_batch']):
                x = np.random.random(dim_x)
                x = cm.scale(x, params)
                x_bounded = []
                for i in range(0,len(x)):
                    elem_bounded = min(x[i],params["max"][i])
                    elem_bounded = max(elem_bounded,params["min"][i])
                    x_bounded.append(elem_bounded)
                n = np.random.randint(0, c.shape[0])
                to_evaluate += [(np.array(x_bounded), f, tasks[n], params)]
                to_evaluate_centroid += [c[n,:]]
            if params['parallel'] == True:
                s_list = pool.map(evaluate, to_evaluate)
            else:
                s_list = map(evaluate, to_evaluate)
            evals += len(to_evaluate)
            b_evals += len(to_evaluate)
            for i in range(0, len(list(s_list))):
                s = cm.__add_to_archive(s_list[i], to_evaluate_centroid[i], archive, kdt)
            init_count = len(archive)
        else:  # variation/selection loop
            keys = list(archive.keys())
            for n in range(0, params['batch_size']):
                # parent selection
                x = archive[keys[np.random.randint(len(keys))]]
                y = archive[keys[np.random.randint(len(keys))]]
                # copy & add variation
                z = cm.variation_xy(x.x, y.x, params)
                # different modes for multi-task (to select the niche)
                to_eval, to_eval_c = select_niche(x, z, f, c, tasks, t_size, params)
                to_evaluate += to_eval
                to_evaluate_centroid += to_eval_c
            # parallel evaluation of the fitness
            if params['parallel'] == True:
                s_list = pool.map(evaluate, to_evaluate)
            else:
                s_list = map(evaluate, to_evaluate)
            evals += len(to_evaluate)
            b_evals += len(to_evaluate)
            # natural selection
            suc = 0
            for i in range(0, len(list(s_list))):
                suc += cm.__add_to_archive(s_list[i], to_evaluate_centroid[i], archive, kdt)
            if params['multi_mode'] == 'tournament_random' or params['multi_mode'] == 'tournament_gp':
                successes[t_size] += [(suc / params["batch_size"], evals)]
        if params['multi_mode'] == 'bandit_niche':
            t_size = opt_tsize(successes, n_niches)
        else:
            t_size = params['n_size']
        # write archive
        if params['dump_period'] != -1 and b_evals > params['dump_period']:
            cm.__save_archive(archive, evals)
            b_evals = 0
            n_e = []
            for v in successes.values():
                n_e += [len(v)]
            print(evals, n_e)
            np.savetxt('t_size.dat', np.array(n_e))
        if log_file != None:
            fit_list = np.array([x.fitness for x in archive.values()])
            log_file.write("{} {} {} {}\n".format(evals, len(archive.keys()), fit_list.max(), fit_list.mean()))
            log_file.flush()
    cm.__save_archive(archive, evals)
    return archive


# a small test
if __name__ == "__main__":
    def rastrigin(xx):
        x = xx * 10.0 - 5.0
        f = 10 * x.shape[0]
        for i in range(0, x.shape[0]):
            f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
        return -f, np.array([xx[0], xx[1]])
    # CVT-based version
    my_map = compute(dim_map=2, dim_x = 10, n_niches=1500, f=rastrigin)
