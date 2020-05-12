import sys, os
import numpy
sys.path.append('./examples')
import hexa_multitask_v as hexa_sim
if __name__ == "__main__":
    optimize_coefficients = False 
    perfect_models = False 
    result_filename = "results.dat2"
    if len(sys.argv) == 1 or ('help' in sys.argv):
        print("Usage: \"python3 ./examples/multitask_arm.py field result_filename optimize_coefficients perfect_models\"")
        exit(0)
    elif len(sys.argv) == 2:
        print("Usage: \"python3 ./examples/multitask_arm.py field result_filename optimize_coefficients perfect_models\"")
        field_to_test = sys.argv[1]
    elif len(sys.argv) == 3:
        print("Usage: \"python3 ./examples/multitask_arm.py field result_filename optimize_coefficients perfect_models\"")
        field_to_test = sys.argv[1]
        result_filename = sys.argv[2]
    elif len(sys.argv) == 4:
        print("Usage: \"python3 ./examples/multitask_arm.py field result_filename optimize_coefficients perfect_models\"")
        field_to_test = sys.argv[1]
        result_filename = sys.argv[2]
        optimize_coefficients = bool(sys.argv[3])
    elif len(sys.argv) == 5:
        field_to_test = sys.argv[1]
        result_filename = sys.argv[2]
        optimize_coefficients = (sys.argv[3] == "True")
        perfect_models = (sys.argv[4] == "True")
    print("Field tested: ", field_to_test)
    print("Optimize coefficients: ", optimize_coefficients)
    print("Perfect Models:", perfect_models)
    print("Result_filename:", result_filename)
    res = []
    n_samples = 20
    for noise_level in [200, 80, 50, 20, 15, 10, 8, 6, 4, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]:
        fitness_list = hexa_sim.fitness_sensitivity(n_samples, [field_to_test], optimize_coefficients, perfect_models, noise_level)
        fitness_mean = numpy.mean(fitness_list)
        fitness_var = numpy.var(fitness_list)
        res.append({'noise_level': noise_level, 'fitness_mean': fitness_mean, 'fitness_var': fitness_var, 'fitness_list': fitness_list})
    result_file = open(result_filename, 'w')
    result_file.write("field_tested: " + field_to_test + "; optimize_coefficients: " + str(optimize_coefficients) + '; perfect_models: ' + str(perfect_models) + '; n_samples: ' + str(n_samples) + '\n')
    for point in res:
        result_file.write(str(point['noise_level']) + ";" + str(point['fitness_mean']) + ";" + str(point['fitness_var']) + "\n")
        result_file.write(str([v.item() for v in point['fitness_list']]) + '\n')
    result_file.close()


