import sys
import kinematic_arm
import numpy as np
import cma
import math

def arm(angles, features):
    angular_range = features[0] / len(angles)
    lengths = np.ones(len(angles)) * features[1] / len(angles)
    target = 0.5 * np.ones(2)
    a = kinematic_arm.Arm(lengths)
    # command in 
    command = (angles - 0.5) * angular_range * math.pi * 2
    ef, _ = a.fw_kinematics(command)
    f = np.linalg.norm(ef - target)
    return f, features

def write_array(a, f):
    for i in a:
        f.write(str(i) + ' ')

# cma
def test_cma(centroids_fname, dim):
    centroids = np.loadtxt(centroids_fname)

    opts = cma.CMAOptions()
    #for i in opts:
    #    print(i, ' => ', opts[i])
    max_evals = 1e6
    opts.set('tolfun', 1e-20)
    opts['tolx'] = 1e-20
    opts['verb_disp'] = 1e10
    opts['maxfevals'] = max_evals / centroids.shape[0]
    opts['BoundaryHandler'] = cma.BoundPenalty
    opts['bounds'] = [0, 1]
    
    es_vector = []
    for c in range(0, centroids.shape[0]):
        es_vector += [cma.CMAEvolutionStrategy(dim * [0.5], 0.5, opts)]

    total_evals = 0
    log = open('cover_max_mean.dat', 'w')
    while total_evals < max_evals:
        result_file = open('archive_'+ str(total_evals) + '.dat', 'w')
        archive = []
        for c in range(0, centroids.shape[0]):
            centroid = centroids[c, :]
            def func(angles):
                return arm(angles, centroid)[0]
            solutions = es_vector[c].ask()
            es_vector[c].tell(solutions, [func(x) for x in solutions])
            total_evals += len(solutions)
            # save to file
            xopt = es_vector[c].result[0]
            xval = es_vector[c].result[1]
            # save
            archive += [-xval]
            # write
            result_file.write(str(-xval) + ' ')
            write_array(centroid, result_file)
            write_array(xopt, result_file)
            result_file.write('\n')
        mean = np.mean(archive)
        max_v = max(archive)
        coverage = len(archive)
        log.write(str(total_evals) + ' ' + str(coverage) + ' ' + str(max_v) + ' ' + str(mean) + '\n')
        log.flush()
        print(total_evals)

test_cma(sys.argv[1], int(sys.argv[2]))
