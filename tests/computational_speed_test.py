import numpy as np
import frictionModelsCPP.build.FrictionModelCPPClass as cpp
import frictionModelsCPP.build.ReducedFrictionModelCPPClass as red_cpp
from tqdm import tqdm
import time


dt = 1e-4
fn = 1
sim_time = 5
N = 10
contact_size = 0.02
n_grids = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 101]
num_time_steps = int(sim_time/dt)

full_model = np.zeros((N, len(n_grids)))
red_model = np.zeros((N, len(n_grids)))
red_pre = np.zeros((N, len(n_grids)))


def properties_to_list(prop):
    list_ = []
    for index, key in enumerate(prop):
        if key == "grid_shape":
            list_.append(prop[key][0])
            list_.append(prop[key][1])
        else:
            list_.append(prop[key])
    return list_

for iii in range(N):
    for i_n_grid, n_grid in enumerate(n_grids):
        p = {'grid_shape': (n_grid, n_grid),  # number of grid elements in x any
             'grid_size': contact_size / n_grid,  # the physical size of each grid element
             'mu_c': 1,
             'mu_s': 1.2,
             'v_s': 1e-3,
             'alpha': 2,
             's0': 1e5,
             's1': 2e1,
             's2': 0.2,
             'dt': dt,
             'z_ba_ratio': 0.9,
             'stability': True,
             'elasto_plastic': True,
             'steady_state': False,
             'n_ls': 20}

        fic = cpp.FullFrictionModel()
        fic.init(properties_to_list(p), "Square", fn)
        start_time = time.time()
        for i_t in range(num_time_steps):
            a = fic.step([0.1, -0.05, 1.59])
        end_time = time.time()
        elapsed_time = end_time - start_time
        full_model[iii, i_n_grid] = num_time_steps/elapsed_time


        start_time = time.time()
        fic_red = red_cpp.ReducedFrictionModel()
        fic_red.init(properties_to_list(p), "Square", fn)
        end_time = time.time()
        elapsed_time = end_time - start_time
        red_pre[iii, i_n_grid] = elapsed_time

        start_time = time.time()
        for i_t in range(num_time_steps):
            a = fic_red.step([0.1, -0.05, 1.59])
        end_time = time.time()
        elapsed_time = end_time - start_time
        red_model[iii, i_n_grid] = num_time_steps/elapsed_time


print('full_model', np.mean(full_model, axis=0))

print('red_pre', np.mean(red_pre, axis=0))

print('red_model', np.mean(red_model, axis=0))