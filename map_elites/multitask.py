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


def add_to_archive(s, archive):
    centroid = cm.make_hashable(s.centroid)
    if centroid in archive:
        if s.fitness > archive[centroid].fitness:
            archive[centroid] = s
            return 1
        return 0
    else:
        archive[centroid] = s
        return 1

# evaluate a single vector (z) with a function f and return a species
# t = vector, function
def __evaluate(t):
    z, f, task, centroid, _ = t
    fit = f(z, task)
    return cm.Species(z, task, fit, centroid)

# bandit opt for optimizing tournament size
# probability matching / Adaptive pursuit Thierens GECCO 2005
# UCB: schoenauer / Sebag
# TODO : params for values, and params for window
def bandit(successes, n_niches):
    n = 0
    for v in successes.values():
        n += len(v)
    v = [1, 10, 50, 100, 500]#, 1000]
    if len(successes.keys()) < len(v):
        return random.choice(v)
    ucb = []
    for k in v:
        x = [i[0] for i in successes[k]]
        mean = sum(x) / float(len(x)) # 100 = batch size??
        n_a = len(x)
        ucb += [mean +  math.sqrt(2 * math.log(n) / n_a)]
    a = np.argmax(ucb)
    t_size = v[a]
    return t_size

# select the niche according to
def select_niche(x, z, f, centroids, tasks, t_size, params, use_distance=False):
    to_evaluate = []
    if not use_distance:
        # No distance: evaluate on a random niche
        niche = np.random.randint(len(tasks))
        to_evaluate += [(z, f, tasks[niche], centroids[niche, :], params)]
    else:
        # we select the parent (a single one), then we select the niche
        # with a tournament based on the task distance
        # the size of the tournament depends on the bandit algorithm
        niches_centroids = []
        niches_tasks = [] # TODO : use a kd-tree
        rand = np.random.randint(centroids.shape[0], size=t_size)
        for p in range(0, t_size):
            n = rand[p]
            niches_centroids += [centroids[n, :]]
            niches_tasks += [tasks[n]]
        cd = distance.cdist(niches_centroids, [x.centroid], 'euclidean')
        cd_min = np.argmin(cd)
        to_evaluate += [(z, f, niches_tasks[cd_min], niches_centroids[cd_min], params)]
    return to_evaluate


def compute(dim_map=-1,
            dim_x=-1,
            f=None,
            max_evals=1e5,
            centroids=[],
            tasks=[],
            variation_operator=cm.variation,
            params=cm.default_params,
            log_file=None):
    """Multi-task MAP-Elites
    - if there is no centroid : random assignation of niches
    - if there is no task: use the centroids as tasks
    - if there is a centroid list: use the centroids to compute distances
    when using the distance, use the bandit to select the tournament size (cf paper):

    Format of the logfile: evals archive_size max mean 5%_percentile, 95%_percentile

    Reference:
    Mouret and Maguire (2020). Quality Diversity for Multitask Optimization
    Proceedings of ACM GECCO.
    """
    print(params)
    assert(f != None)
    assert(dim_x != -1)
    # handle the arguments
    use_distance = False
    if tasks != [] and centroids != []:
        use_distance = True
    elif tasks == [] and centroids != []:
        # if no task, we use the centroids as tasks
        tasks = centroids
        use_distance = True
    elif tasks != [] and centroids == []:
        # if no centroid, we create indices so that we can index the archive by centroid
        centroids = np.arange(0, len(tasks)).reshape(len(tasks), 1)
        use_distance = False
    else:
        raise ValueError('Multi-task MAP-Elites: you need to specify a list of task, a list of centroids, or both')
    print("Multitask-MAP-Elites:: using distance =>", use_distance)

    assert(len(tasks) == len(centroids))
    n_tasks = len(tasks)

    # init archive (empty)
    archive = {}

    init_count = 0

    # init multiprocessing
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # main loop
    n_evals = 0 # number of evaluations
    b_evals = 0 # number evaluation since the last dump
    t_size = 1  # size of the tournament (if using distance) [will be selected by the bandit]
    successes = defaultdict(list) # count the successes
    while (n_evals < max_evals):
        to_evaluate = []
        to_evaluate_centroid = []
        if len(archive) <= params['random_init'] * n_tasks:
            # initialize the map with random individuals
            for i in range(0, params['random_init_batch']):
                # create a random individual
                x = np.random.uniform(low=params['min'], high=params['max'], size=dim_x)
                # we take a random task
                n = np.random.randint(0, n_tasks)
                to_evaluate += [(x, f, tasks[n], centroids[n], params)]
            s_list = cm.parallel_eval(__evaluate, to_evaluate, pool, params)
            n_evals += len(to_evaluate)
            b_evals += len(to_evaluate)
            for i in range(0, len(list(s_list))):
                add_to_archive(s_list[i], archive)
        else:
            # main variation/selection loop
            keys = list(archive.keys())
            # we do all the randint together because randint is slow
            rand1 = np.random.randint(len(keys), size=params['batch_size'])
            rand2 = np.random.randint(len(keys), size=params['batch_size'])
            for n in range(0, params['batch_size']):
                # parent selection
                x = archive[keys[rand1[n]]]
                y = archive[keys[rand2[n]]]
                # copy & add variation
                z = variation_operator(x.x, y.x, params)
                # different modes for multi-task (to select the niche)
                to_evaluate += select_niche(x, z, f, centroids, tasks, t_size, params, use_distance)
            # parallel evaluation of the fitness
            s_list = cm.parallel_eval(__evaluate, to_evaluate, pool, params)
            n_evals += len(to_evaluate)
            b_evals += len(to_evaluate)
            # natural selection
            suc = 0
            for i in range(0, len(list(s_list))):
                suc += add_to_archive(s_list[i], archive)
            if use_distance:
                successes[t_size] += [(suc, n_evals)]
        if use_distance: # call the bandit to optimize t_size
            t_size = bandit(successes, n_tasks)

        # write archive
        if params['dump_period'] != -1 and b_evals > params['dump_period']:
            cm.__save_archive(archive, n_evals)
            b_evals = 0
            n_e = [len(v) for v in successes.values()]
            print(n_evals, n_e)
            np.savetxt('t_size.dat', np.array(n_e))
        if log_file != None:
            fit_list = np.array([x.fitness for x in archive.values()])
            log_file.write("{} {} {} {} {} {} {}\n".format(n_evals, len(archive.keys()), fit_list.max(), np.mean(fit_list), np.median(fit_list), np.percentile(fit_list, 5), np.percentile(fit_list, 95)))
            log_file.flush()
    cm.__save_archive(archive, n_evals)
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
