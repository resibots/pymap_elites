# only required to run python3 examples/cvt_rastrigin.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import map_elites.cvt as cvt_map_elites
import numpy as np
import math

def rastrigin(xx):
    x = xx * 10.0 - 5.0
    f = 10 * x.shape[0]
    for i in range(0, x.shape[0]):
        f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
    return -f, np.array([xx[0], xx[1]])

archive = cvt_map_elites.compute(2, 6, rastrigin, n_niches=5000, n_gen=2500, log_file=open('cover_max_mean.dat', 'w'))
