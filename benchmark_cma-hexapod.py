import sys
import numpy as np
import cma
import math
import pyhexapod.simulator as simulator
import pycontrollers.hexapod_controller as ctrl
import pybullet
import time
import multiprocessing


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


def write_array(a, f):
    for i in a:
        f.write(str(i) + ' ')

def load(directory, k):
    tasks = []
    centroids = []
    for i in range(0, k):
        centroid = np.loadtxt(directory + '/lengthes_' + str(i) + '.txt')
        urdf_file = directory + '/pexod_' + str(i) + '.urdf'
        centroids += [centroid]
        tasks += [(centroid, urdf_file)]
    return np.array(centroids), tasks

def evaluate(y):
    x, task, func = y
    return func(x, task)[0]

# cma
def test_cma(urdf_directory, dim):

    print('loading files...', end='')
    centroids, tasks = load(urdf_directory, 2)
    print('data loaded')
    
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

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    total_evals = 0
    log = open('cover_max_mean.dat', 'w')
    while total_evals < max_evals:
        #result_file = open('archive_'+ str(total_evals) + '.dat', 'w')
        archive = []
        for c in range(0, centroids.shape[0]):
            centroid = centroids[c, :]
            task = tasks[c]
            def func(angles):
                return hexapod(angles, task)[0]
            solutions = es_vector[c].ask()
#            print(len(solutions))# pop =14
            s_list = pool.map(evaluate, [(x, task, hexapod) for x in solutions])
            es_vector[c].tell(solutions, s_list)
            total_evals += len(solutions)
            # save to file
            xopt = es_vector[c].result[0]
            xval = es_vector[c].result[1]
            # save
            archive += [-xval]
            # write
            #result_file.write(str(-xval) + ' ')
            #write_array(centroid, result_file)
            #write_array(xopt, result_file)
            #result_file.write('\n')
        mean = np.mean(archive)
        max_v = max(archive)
        coverage = len(archive)
        log.write(str(total_evals) + ' ' + str(coverage) + ' ' + str(max_v) + ' ' + str(mean) + '\n')
        log.flush()
        print(total_evals)

test_cma(sys.argv[1], 36)
