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

import math
import numpy as np
import multiprocessing

# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

from map_elites import common as cm

# evaluate a single vector (x) with a function f and return a species
# t = vector, function
def evaluate(t):
    z, f = t  # evaluate z with function f
    fit, desc = f(z)
    return cm.Species(z, desc, fit)

# map-elites algorithm (CVT variant)
def compute(dim_map, dim_x, f, n_niches=1000, n_gen=1000, params=cm.default_params, log_file=None):
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # create the CVT
    c = cm.__cvt(n_niches, dim_map,
              params['cvt_samples'], params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    cm.__write_centroids(c)

    # init archive (empty)
    archive = {}

    init_count = 0
    evals = 0
    # main loop
    for g in range(0, n_gen + 1):
        to_evaluate = []
        if g == 0:  # random initialization
            while(init_count<=params['random_init'] * n_niches):
                for i in range(0, params['random_init_batch']):
                    x = np.random.random(dim_x)
                    x = cm.scale(x, params)
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
                evals += len(to_evaluate)
                for s in s_list:
                    cm.__add_to_archive(s, s.desc, archive, kdt)
                init_count = len(archive)
                to_evaluate = []
        else:  # variation/selection loop
            keys = list(archive.keys())
            for n in range(0, params['batch_size']):
                # parent selection
                x = archive[keys[np.random.randint(len(keys))]]
                # copy & add variation
                z = cm.variation(x.x, archive, params)
                to_evaluate += [(z, f)]
            # parallel evaluation of the fitness
            if params['parallel'] == True:
                s_list = pool.map(evaluate, to_evaluate)
            else:
                s_list = map(evaluate, to_evaluate)
            evals += len(to_evaluate)
            # natural selection
            for s in s_list:
                cm.__add_to_archive(s, s.desc, archive, kdt)
        # write archive
        if g % params['dump_period'] == 0 and params['dump_period'] != -1:
            print("generation:", g)
            cm.__save_archive(archive, g)
        # write log
        if log_file != None:
            fit_list = np.array([x.fitness for x in archive.values()])
            log_file.write("{} {} {} {}\n".format(evals, len(archive.keys()), fit_list.max(), fit_list.mean()))
            log_file.flush()
        cm.__save_archive(archive, n_gen)
    return archive


