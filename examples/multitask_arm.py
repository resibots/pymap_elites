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

import kinematic_arm
import math
import numpy as np
import sys

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import map_elites.multitask as mt_map_elites
import map_elites.common as cm_map_elites

def arm(angles, task):
    angular_range = task[0] / len(angles)
    lengths = np.ones(len(angles)) * task[1] / len(angles)
    target = 0.5 * np.ones(2)
    a = kinematic_arm.Arm(lengths)
    # command in
    command = (angles - 0.5) * angular_range * math.pi * 2
    ef, _ = a.fw_kinematics(command)
    f = -np.linalg.norm(ef - target)
    return f


if len(sys.argv) == 1 or ('help' in sys.argv):
    print("Usage: \"python3 ./examples/multitask_arm.py 10 [no_distance]\"")
    exit(0)


dim_x = int(sys.argv[1])

# dim_map, dim_x, function
px = cm_map_elites.default_params.copy()
px["dump_period"] = 2000
px["parallel"] = False

n_tasks = 5000
dim_map = 2
# example : create centroids using a CVT
c = cm_map_elites.cvt(n_tasks, dim_map, 30000, True)

# CVT-based version
if len(sys.argv) == 2 or sys.argv[2] == 'distance':
    archive = mt_map_elites.compute(dim_x = dim_x, f=arm, centroids=c, max_evals=1e6, params=px, log_file=open('mt_dist.dat', 'w'))
else:
    # no distance:
    archive = mt_map_elites.compute(dim_x = dim_x, f=arm, tasks=c, max_evals=1e6, params=px, log_file=open('mt_no_dist.dat', 'w'))
