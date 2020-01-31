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
import math
import numpy as np
import sys
import pyhexapod.simulator as simulator
import pycontrollers.hexapod_controller as ctrl
import pybullet
import time


def hexapod(x, features):
    t0 = time.perf_counter()
    urdf_file = features[1]
    simu = simulator.HexapodSimulator(gui=False, urdf=urdf_file)
    controller = ctrl.HexapodController(x)
    dead = False
    fit = -1e10
    steps = 3. / simu.dt
    i = 0
    while i < steps and not dead:
        simu.step(controller)
        p = simu.get_pos()[0] 
        a = pybullet.getEulerFromQuaternion(simu.get_pos()[1])
        out_of_corridor = abs(p[1]) > 0.5
        out_of_angles = abs(a[0]) > math.pi/8 or abs(a[1]) > math.pi/8 or abs(a[2]) > math.pi/8
        if out_of_angles or out_of_corridor:
            dead = True
        i += 1
    fit = p[0]
    #print(time.perf_counter() - t0, " ms", '=>', fit)
    return fit, features    


def load(directory, k):
    tasks = []
    centroids = []
    for i in range(0, k):
        centroid = np.loadtxt(directory + '/lengthes_' + str(i) + '.txt')
        urdf_file = directory + '/pexod_' + str(i) + '.urdf'
        centroids += [centroid]
        tasks += [(centroid, urdf_file)]
    return np.array(centroids), tasks

print('loading files...', end='')
centroids, tasks = load(sys.argv[2], 2000)
print('data loaded')
dim_x = 36

px = map_elites.default_params.copy()
px['multi_task'] = True
px['multi_mode'] = sys.argv[1]
px['n_size'] = 100
px['min'] = [0.] * dim_x
px['max'] = [1.] * dim_x


archive = map_elites.compute(dim_map=2, dim_x=dim_x, f=hexapod, centroids=centroids, tasks=tasks, num_evals=1e6, params=px, log_file=open('cover_max_mean.dat', 'w'))
