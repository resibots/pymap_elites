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

#    f.write(str(k.fitness) + ' ')
# write_array(k.centroid, f)
# write_array(k.desc, f)
# write_array(k.x, f)

                                    
def load_controllers(fname):
    f = open(fname)
    all_lines = f.readlines()
    fit = []
    gen = []
    centroids = []
    urdfs = []
    for i in range(0, len(all_lines), 2):
        r = all_lines[i] + ' '  + all_lines[i + 1]
        r_s = r.split(' ')
        for j in r_s:
            if 'urdf' in j:
                urdf = j
        urdfs += [urdf]
        fitness = float(r_s[0])
        centroid = r_s[1:13]
        x_str = r_s[len(r_s) - 36 - 1:len(r_s) -1] # \n
        assert(len(x_str) == 36)
        assert(len(centroid) == 12)
        fit += [float(fitness)]
        gen += [[float(x) for x in x_str]]
        centroids += [centroid]
    return fit, gen, centroids, urdfs

print('loading URDF/centroids files...', end='')
centroids, tasks = load(sys.argv[1], 2000)
print('data loaded')
dim_x = 36

print('loading controllers from archive:', sys.argv[2])
fit, gen, centroids, urdfs = load_controllers(sys.argv[2])
print('done')
##assert(len(fit) == len(centroids))
#assert(len(fit) == len(gen))
#assert(len(fit) == len(tasks))

for i in range(0, len(fit)):
    #print(gen[i])
    ff,_ = hexapod(gen[i], (centroids[i], urdfs[i]))
    print(ff,fit[i])
