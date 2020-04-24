import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('../DroneLearning')
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


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

torch.set_default_dtype(torch.float64)  # default type for tensors is double

model = HexaVectorized.Hexarotor()
n_models = 5000
default_parameters = HexaVectorized.default_parameters()
default_mixing_matrix, _, _, _, _ = HexaVectorized.gen_matrices_and_other([default_parameters], model)
default_mixing_matrix = torch.squeeze(default_mixing_matrix, dim=0)
default_inv_mixing_matrix = default_mixing_matrix.inverse()
# print("policy matrix shape", default_inv_mixing_matrix.shape)
policy = Debug_HexaV(default_inv_mixing_matrix)
dt = 2e-3
T = int(math.floor(1./dt * 5.0))
dim_model = model.n_state
init_state = torch.zeros((1, dim_model))
axes = torch.randn(1, 3) * torch.tensor((0.05, -0.05, 1.0))
angles = Quaternions.axis_angle_to_quaternion(axes)
init_state[:, 6:10] = init_state[:, 6:10] + angles.squeeze(dim=0)
setpoints = torch.zeros((T, 1, dim_model))
setpoints[:, :, 6] = 1
init_state2 = torch.zeros((1, model.n_state))
init_state2[:, 6] = 1
init_state2[:, 4] = 1
init_state2[:, 3] = 1
setpoints2 = torch.zeros((T, 1, model.n_state))
setpoints2[:, :, 6] = 1
setpoints2[:, :, 2] = 1

def hexa(coeffs, tasks):
    mixing_matrices = torch.cat(tuple([t[0].unsqueeze(dim =0) for t in tasks]))
    m = torch.cat(tuple([t[1].unsqueeze(dim =0) for t in tasks]))
    inertia = torch.cat(tuple([t[2].unsqueeze(dim =0) for t in tasks]))
    inv_inertia = torch.cat(tuple([t[3].unsqueeze(dim =0) for t in tasks]))
    motor_total_change_time = torch.cat(tuple([t[4].unsqueeze(dim = 0) for t in tasks]))
    coeffs = torch.cat(tuple([torch.tensor(c).unsqueeze(dim = 0) for c in coeffs]))
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
    state_history = run_traj_parallel_hexa(policy, model, init_state.repeat((n,1)), setpoints.repeat((1,n,1)), T, dt, m, inertia, inv_inertia, mixing_matrices, motor_total_change_time, coeffs)
    f = -model.cost_from_trajectory(state_history, setpoints).mean(dim = 0)
    state_history2 = run_traj_parallel_hexa(policy, model, init_state2.repeat((n,1)), setpoints2.repeat((1,n,1)), T, dt, m, inertia, inv_inertia, mixing_matrices, motor_total_change_time, coeffs)
    f -= model.cost_from_trajectory(state_history2, setpoints2).mean(dim = 0)
    return f

models_params = [HexaVectorized.noisy_parameters() for i in range(n_models)]
# models_params = [HexaVectorized.default_parameters() for i in range(n_models)]
# print([a.keys() for a in models_params])
# print([[(k,v.shape) for k,v in a.items()] for a in models_params])
# print([(k,v.shape) for k,v in models_params[0].items()])
# models_params, seeds = read_params_from_archive("archive_3003224.dat")
seeds = None
# print([(k,v.shape) for k,v in models_params[0].items()])
mixing_matrices, m, inertia, inv_inertia, motor_total_change_time = HexaVectorized.gen_matrices_and_other(models_params, model)
# print("building tasks")
# print(mixing_matrices.shape)
# print(m.shape)
# print(inv_inertia.shape)
# print(inertia.shape)
tasks = [(mixing_matrices[i], m[i], inertia[i], inv_inertia[i], motor_total_change_time[i]) for i in range(n_models)]
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
archive = mt_map_elites.compute(dim_map=dim_map, dim_x=dim_x, centroids=centroids, tasks = tasks, f=hexa, max_evals=1e5, params=px, log_file=open('cover_max_mean.dat', 'w'), seeds =  seeds)
# import cma
# es = cma.CMAEvolutionStrategy(dim_x * [0.5], 1)
# for i in range(500):
#     print("Gen:", i, "--------------------------------------------------------------------------")
#     X = es.ask()
#     task = [tasks[0] for _ in range(len(X))]
#     fitness = hexa(X, task)
#     es.tell(X, [-max(fitness[i].item(), -1e50) for i in range(len(X))])
#     # es.disp()
#     # for i in range(len(X)):
#     #     print(" --- fitness;", max(fitness[i].item(), -1e50)," --- ")
#     #     print("PX:", X[i][0],"IX:", X[i][3],"DX:", X[i][6], "PY:", X[i][1],"IY:", X[i][4],"DY:", X[i][7],"PZ:", X[i][2],"IZ:", X[i][5],"DZ:", X[i][8])
#     #     print("PAX:", X[i][9],"IAX:", X[i][12],"DAX:", X[i][15],"PAY:", X[i][10],"IAY:", X[i][13],"DAY:", X[i][16],"PAZ:", X[i][11],"IAZ:", X[i][14],"DAZ:", X[i][17])
#     #     # print([(X[i], max(fitness[i].item(), -1e50)) for i in range(len(X))])
#     print("mean fitness:",  sum([(-max(fitness[i], -1e50)) for i in range(len(X))])/len(X))
#     print("Dist mean:")
#     print("PX:", es.mean[0],"IX:", es.mean[3],"DX:", es.mean[6], "PY:", es.mean[1],"IY:", es.mean[4],"DY:", es.mean[7],"PZ:", es.mean[2],"IZ:", es.mean[5],"DZ:", es.mean[8])
#     print("PAX:", es.mean[9],"IAX:", es.mean[12],"DAX:", es.mean[15],"PAY:", es.mean[10],"IAY:", es.mean[13],"DAY:", es.mean[16],"PAZ:", es.mean[11],"IAZ:", es.mean[14],"DAZ:", es.mean[17])
#     std = es.sigma * es.sigma_vec.scaling * np.sqrt(es.dC) * es.gp.scales
#     print("Dist std:")
#     print("PX:", std[0],"IX:", std[3],"DX:", std[6], "PY:", std[1],"IY:", std[4],"DY:", std[7],"PZ:", std[2],"IZ:", std[5],"DZ:", std[8])
#     print("PAX:", std[9],"IAX:", std[12],"DAX:", std[15],"PAY:", std[10],"IAY:", std[13],"DAY:", std[16],"PAZ:", std[11],"IAZ:", std[14],"DAZ:", std[17])

end = time.time()
print(end - start)
