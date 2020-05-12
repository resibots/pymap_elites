import cma
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('../pytorchdronelearning')
sys.path.append('../')
import map_elites.common as cm_map_elites
import map_elites.multitask as mt_map_elites
from map_elites.read_archive import *
import time
import torch
import numpy as np
import math
import Quaternions
from TrajGeneration import run_traj_parallel_hexa
from Policies import Debug_HexaV
from Model import HexaVectorized
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.set_default_dtype(torch.float64)  # default type for tensors is double
class HexaTasks:
    def __init__(self, large_drone = False, perfect_model = False):
        self.model = HexaVectorized.Hexarotor()
        self.perfect_model = perfect_model
        self.default_parameters = (HexaVectorized.default_parameters(large_drone))
        default_mixing_matrix, _, _, _, _ = HexaVectorized.gen_matrices_and_other([self.default_parameters], self.model)
        default_mixing_matrix = torch.squeeze(default_mixing_matrix, dim=0)
        self.default_mixing_matrix = default_mixing_matrix
        default_inv_mixing_matrix = default_mixing_matrix.inverse()
        self.default_inv_mixing_matrix = default_inv_mixing_matrix
        self.policy = Debug_HexaV(self.default_inv_mixing_matrix)
        self.dt = 2e-3
        self.T = int(math.floor(1./self.dt * 5.0))
        self.dim_model = self.model.n_state
        init_state = torch.zeros((1, self.dim_model))
        init_state[:,6] = 1
        setpoints = torch.zeros((self.T, 1, self.dim_model))
        for t in range(self.T):
            axes = torch.ones(1, 3) * torch.tensor((0.0, 0.0, 0.5 * math.pi * self.dt * t ))
            angles = Quaternions.axis_angle_to_quaternion(axes)
            setpoints[t,:,6:10] = angles
        init_state2 = torch.zeros((1,self.dim_model))
        init_state2[:,6] = 1
        setpoints2 = torch.zeros((self.T, 1, self.dim_model))
        setpoints2[:,:,6] = 1
        for t in range(self.T):
            setpoints2[t,:,2] = 0.08 * t * self.dt
            setpoints2[t,:,1] = 0.1 * math.sin(2*t*self.dt)
            setpoints2[t,:,0] = 0.1 * (math.cos(2*t*self.dt)-1)
        self.init_state = init_state
        self.init_state2 = init_state2
        self.setpoints = setpoints
        self.setpoints2 = setpoints2

    def run(self, coeffs, tasks):
        mixing_matrices = torch.cat(tuple([t[0].unsqueeze(dim =0) for t in tasks]))
        inv_matrices = mixing_matrices.inverse()
        m = torch.cat(tuple([t[1].unsqueeze(dim =0) for t in tasks]))
        inertia = torch.cat(tuple([t[2].unsqueeze(dim =0) for t in tasks]))
        inv_inertia = torch.cat(tuple([t[3].unsqueeze(dim =0) for t in tasks]))
        motor_total_change_time = torch.cat(tuple([t[4].unsqueeze(dim = 0) for t in tasks]))
        coeffs = torch.cat(tuple([c.clone().detach().unsqueeze(dim = 0) for c in coeffs]))
        n = mixing_matrices.shape[0]
        # print(m.shape)
        assert(m.shape[0] == n)
        # print(inertia.shape)
        assert(inertia.shape[0] == n)
        # print(inv_inertia.shape)
        assert(inv_inertia.shape[0] == n)
        # print(motor_total_change_time.shape)
        assert(motor_total_change_time.shape[0] == n)
        # print(coeffs.shape)
        assert(coeffs.shape[0] == n)
        state_history = run_traj_parallel_hexa(self.policy, self.model, self.init_state.repeat((n,1)), self.setpoints.repeat((1,n,1)), self.T, self.dt, m, inertia, inv_inertia, mixing_matrices, motor_total_change_time, coeffs, self.perfect_model)
        f = -self.model.cost_from_trajectory(state_history, self.setpoints).mean(dim = 0)
        state_history2 = run_traj_parallel_hexa(self.policy, self.model, self.init_state2.repeat((n,1)), self.setpoints2.repeat((1,n,1)), self.T, self.dt, m, inertia, inv_inertia, mixing_matrices, motor_total_change_time, coeffs, self.perfect_model)
        f -= self.model.cost_from_trajectory(state_history2, self.setpoints2).mean(dim = 0)
        return f
def CMAES_search(hexa_simu, model_params, task, iterations):
    dim_x = 18
    es = cma.CMAEvolutionStrategy(dim_x * [0.05], 0.1, {'bounds': [0., 1.]})
    for k,v in model_params.items():
        print(k,v)

    min_fitness = float('inf') 
    for i in range(iterations):
        print("Gen:", i, "--------------------------------------------------------------------------")
        X = es.ask()
        task_list = [task for _ in range(len(X))]
        fitness = hexa_simu.run([torch.tensor(x) for x in X], task_list)
        fitness = [-max(fitness[i].item(), -1e50) for i in range(len(X))]
        min_fitness = min(min(fitness), min_fitness)
        es.tell(X, fitness)
        mean_fitness_list = []
        mean_std_list = []
        # es.disp()
        # for i in range(len(X)):
        #     print(" --- fitness;", max(fitness[i].item(), -1e50)," --- ")
        #     print("PX:", X[i][0],"IX:", X[i][3],"DX:", X[i][6], "PY:", X[i][1],"IY:", X[i][4],"DY:", X[i][7],"PZ:", X[i][2],"IZ:", X[i][5],"DZ:", X[i][8])
        #     print("PAX:", X[i][9],"IAX:", X[i][12],"DAX:", X[i][15],"PAY:", X[i][10],"IAY:", X[i][13],"DAY:", X[i][16],"PAZ:", X[i][11],"IAZ:", X[i][14],"DAZ:", X[i][17])
        #     # print([(X[i], max(fitness[i].item(), -1e50)) for i in range(len(X))])
        def print_gen(X, fitness):
            print("fitness:",  fitness)
            mean_fitness = sum(fitness)/len(X)
            print("mean fitness:", mean_fitness)
            mean_fitness_list.append(mean_fitness)
            print("Dist mean:")
            practical_mean = X.mean(axis = 0)
            print("PX:", practical_mean[0],"IX:", practical_mean[3],"DX:", practical_mean[6], "PY:", practical_mean[1],"IY:", practical_mean[4],"DY:", practical_mean[7],"PZ:", practical_mean[2],"IZ:", practical_mean[5],"DZ:", practical_mean[8])
            # print("PX:", es.mean[0],"IX:", es.mean[3],"DX:", es.mean[6], "PY:", es.mean[1],"IY:", es.mean[4],"DY:", es.mean[7],"PZ:", es.mean[2],"IZ:", es.mean[5],"DZ:", es.mean[8])
            print("PAX:", practical_mean[9],"IAX:", practical_mean[12],"DAX:", practical_mean[15],"PAY:", practical_mean[10],"IAY:", practical_mean[13],"DAY:", practical_mean[16],"PAZ:", practical_mean[11],"IAZ:", practical_mean[14],"DAZ:", practical_mean[17])
            # print("PAX:", es.mean[9],"IAX:", es.mean[12],"DAX:", es.mean[15],"PAY:", es.mean[10],"IAY:", es.mean[13],"DAY:", es.mean[16],"PAZ:", es.mean[11],"IAZ:", es.mean[14],"DAZ:", es.mean[17])
            # std = es.sigma * es.sigma_vec.scaling * np.sqrt(es.dC) * es.gp.scales
            practical_std = X.std(axis = 0)
            mean_std_list.append(practical_std.mean())
            print("Dist std:")
            # print("PX:", std[0],"IX:", std[3],"DX:", std[6], "PY:", std[1],"IY:", std[4],"DY:", std[7],"PZ:", std[2],"IZ:", std[5],"DZ:", std[8])
            # print("PAX:", std[9],"IAX:", std[12],"DAX:", std[15],"PAY:", std[10],"IAY:", std[13],"DAY:", std[16],"PAZ:", std[11],"IAZ:", std[14],"DAZ:", std[17])
            print("PX:", practical_std[0],"IX:", practical_std[3],"DX:", practical_std[6], "PY:", practical_std[1],"IY:", practical_std[4],"DY:", practical_std[7],"PZ:", practical_std[2],"IZ:", practical_std[5],"DZ:", practical_std[8])
            print("PAX:", practical_std[9],"IAX:", practical_std[12],"DAX:", practical_std[15],"PAY:", practical_std[10],"IAY:", practical_std[13],"DAY:", practical_std[16],"PAZ:", practical_std[11],"IAZ:", practical_std[14],"DAZ:", practical_std[17])
            print('\n')
            print("min_fitness", min_fitness)
            print('\n')
        print_gen(numpy.array(X),fitness)
    return mean_fitness_list, mean_std_list

def fitness_sensitivity(n_samples, parameter_name_list, optimized_coefficients = False, perfect_model = False, noise_level = 1.0):
    hexa_simu = HexaTasks(large_drone = False, perfect_model = perfect_model)
    models_params = [HexaVectorized.noisy_parameters(noise_fields = parameter_name_list, noise_level = noise_level) for i in range(n_samples)]
    mixing_matrices, m, inertia, inv_inertia, motor_total_change_time = HexaVectorized.gen_matrices_and_other(models_params, hexa_simu.model)
    tasks = [(mixing_matrices[i], m[i], inertia[i], inv_inertia[i], motor_total_change_time[i]) for i in range(len(models_params))]
    if optimized_coefficients:
        fitness_list = []
        for i in range(n_samples):
            mean_f, mean_v = CMAES_search(hexa_simu, models_params[i], tasks[i], iterations = 100)
            fitness_list.append(mean_f[-1])
    else:
        fitness_tensor = hexa_simu.run(torch.ones((n_samples, 18)) * 0.05 , tasks)
        fitness_list = [fitness_tensor[i] for i in range(n_samples)]
    return fitness_list

if __name__ == "__main__":
    large_drone = False
    perfect_model = True
    CMAES_check = False
    n_models = 5000
    hexa_simu = HexaTasks(large_drone, perfect_model)
    seeds = None
    # print([a.keys() for a in models_params])
    # print([[(k,v.shape) for k,v in a.items()] for a in models_params])
    # print([(k,v.shape) for k,v in models_params[0].items()])
    if CMAES_check:
        models_params, seeds = read_params_from_archive("archive_1000024.log")
        for i,s in enumerate(seeds):
            if s.fitness < -1 and s.fitness > -5:
                models_params = [models_params[i]]
                print(s.fitness)
                break
        print(models_params)
    else:
        models_params = [HexaVectorized.noisy_parameters(large_drone) for i in range(n_models)]
        # models_params = [HexaVectorized.default_parameters(large_drone) for i in range(n_models)]
    # print([(k,v.shape) for k,v in models_params[0].items()])
    mixing_matrices, m, inertia, inv_inertia, motor_total_change_time = HexaVectorized.gen_matrices_and_other(models_params, hexa_simu.model)
    # print(mixing_matrices.shape)
    # print(m.shape)
    # print(inv_inertia.shape)
    # print(inertia.shape)
    tasks = [(mixing_matrices[i], m[i], inertia[i], inv_inertia[i], motor_total_change_time[i]) for i in range(len(models_params))]
    centroids = torch.cat(tuple([torch.cat([v for k,v in p.items()]).unsqueeze(dim = 0) for p in models_params])).numpy()
    dim_map = centroids.shape[0]
    dim_x = 18
    # dim_map, dim_x, function
    px = cm_map_elites.default_params.copy()
    px['parallel'] = True
    px['batch_size'] = 128
    # px['parallel'] = False
    px['multi_task'] = True
    px['multi_mode'] = 'bandit_niche'
    px['n_size'] = 0  # ignored if bandit_niche
    px["dump_period"] = 10000
    px["min"] = 0. 
    px["max"] = 1.

    # CVT-based version
    start = time.time()
    if CMAES_check:
        mean_f, mean_v = CMAES_search(hexa_simu, models_params[0], tasks[0], 1000)
    else:
        archive = mt_map_elites.compute(dim_map=dim_map, dim_x=dim_x, centroids=centroids, tasks = tasks, f=hexa_simu.run, max_evals=1e6, params=px, log_file=open('cover_max_mean.dat', 'w'), seeds =  seeds)
    end = time.time()
    print(end - start)
