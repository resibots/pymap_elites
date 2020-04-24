import numpy
import math
import seaborn as sns
import matplotlib.pyplot as plt
from map_elites.common import Species
import torch
def to_np_array(l):
    return numpy.array([numpy.array(x) for x in l])
def read_from_archive(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    archive = []
    for l in lines:
        line_s = l.split(' ')
        fitness = float(line_s[0])
        centroid = to_np_array([float(i) for i in line_s[1:-19]])
        x = to_np_array([float(i) for i in line_s[-19:-1]])
        # print(len(centroid), len(x))
        # print(x, centroid)
        archive.append({'x': x, "fitness" : fitness, "centroid" : centroid})
    file.close()
    return archive
def array_to_param_dict(array):
    mass = array[0]
    inertia = array[1:4]
    k_t = array[4:10]
    k_d = array[10:16]
    alpha = array[16:22]
    beta = array[22:28]
    motor = array[43:49]
    theta = array[28:34]
    delta_G = array[34:37]
    l = array[37:43]
    return {
        'mass' : torch.tensor([mass]),
        'diag_inertia' : torch.tensor(inertia),
        'k_t': torch.tensor(k_t),
        'k_d': torch.tensor(k_d),
        'alpha' : torch.tensor(alpha),
        'beta' : torch.tensor(beta),
        'theta' : torch.tensor(theta),
        'delta_G' : torch.tensor(delta_G),
        'l' : torch.tensor(l),
        'motor_total_change_time' : torch.tensor(motor),
    }
def read_params_from_archive(filename):
    archive = read_from_archive(filename)
    model_params = [array_to_param_dict(a['centroid']) for a in archive]
    s_list = [Species(a['x'], a['centroid'], a['fitness'],a['centroid']) for a in archive]
    return model_params, s_list