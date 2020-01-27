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

import map_elites
import kinematic_arm
import math
import numpy as np
import sys

def arm(angles, features):
    angular_range = features[0] / len(angles)
    lengths = np.ones(len(angles)) * features[1] / len(angles)
    target = 0.5 * np.ones(2)
    a = kinematic_arm.Arm(lengths)
    # command in 
    command = (angles - 0.5) * angular_range * math.pi * 2
    ef, _ = a.fw_kinematics(command)
    f = -np.linalg.norm(ef - target)
    return f, features

dim_x = int(sys.argv[3])

# dim_map, dim_x, function
# archive = compute(dim_map=2, dim_x=6, f=rastrigin, n_niches=5000, n_gen=2500)
px = map_elites.default_params.copy()
px['multi_task'] = True
px['multi_mode'] = sys.argv[1]
px['n_size'] = int(sys.argv[2])
px["dump_period"] = 2000
px["min"] = [0.]*dim_x
px["max"] = [1.]*dim_x



# CVT-based version
archive = map_elites.compute(dim_map=2, dim_x=dim_x, f=arm, n_niches=5000, num_evals=1e6, params=px, log_file=open('cover_max_mean.dat', 'w'))

# task-based version (random centroids)
#tasks = np.random.random((1000, 2))
#centroids = tasks
#archive = map_elites.compute(dim_map=2, dim_x=dim_x, f=arm, centroids=centroids, tasks=tasks, num_evals=2e5, params=px, log_file='evals_cover_max_mean.dat')
