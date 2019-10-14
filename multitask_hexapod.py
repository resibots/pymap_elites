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

def hexapod(param, features):
    simu = simulator.HexapodSimulator(gui=False)
    controller = ctrl.HexapodController(ctrl)
    for i in range(0, int(3./simu.dt)): # seconds
        simu.step(controller)
    return f, features

# dim_map, dim_x, function
# archive = compute(dim_map=2, dim_x=6, f=rastrigin, n_niches=5000, n_gen=2500)
px = map_elites.default_params.copy()
px['multi_task'] = True
px['multi_mode'] = sys.argv[1]
px['n_size'] = int(sys.argv[2])
dim_x = int(sys.argv[3])
archive = map_elites.compute(dim_map=2, dim_x=dim_x, f=hexapod, n_niches=1000, num_evals=2e5, params=px)
