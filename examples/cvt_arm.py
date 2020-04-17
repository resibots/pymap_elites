# only required to run python3 examples/cvt_arm.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import math

import map_elites.cvt as cvt_map_elites
import map_elites.common as cm_map_elites
import kinematic_arm


def arm(angles):
    lengths = np.ones(len(angles)) / len(angles)
    a = kinematic_arm.Arm(lengths)
    command = (angles - 0.5) * math.pi * 2
    ef, _ = a.fw_kinematics(command)
    f = -np.std(command) # fitness
    desc = ef / 2. + 0.5 # descriptor (position) in [0, 1]
    return f, desc


px = cm_map_elites.default_params.copy()

archive = cvt_map_elites.compute(2, 5, arm, n_niches=10000, max_evals=1e6, log_file=open('cvt_arm.dat', 'w'), params=px)
