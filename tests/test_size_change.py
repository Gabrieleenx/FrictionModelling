import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from velocity_profiles import vel_num_cells
import surfaces.surfaces as surf
import matplotlib as mpl
from frictionModels.frictionModel import ReducedFrictionModel
from frictionModels.utils import vel_to_cop

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']

properties = {'grid_shape': (20, 20),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.3,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e5,
              's1': 2e1,
              's2': 0.4,
              'dt': 1e-4,
              'z_ba_ratio': 0.9,
              'stability': False,
              'elasto_plastic': True,
              'steady_state': True}
vel = {'x':0.001, 'y':0, 'tau':1}

shape = surf.PObject(properties['grid_size'], properties['grid_shape'], surf.p_square)

planar_lugre_reduced = ReducedFrictionModel(properties=properties)
planar_lugre_reduced.update_p_x_y(shape)
planar_lugre_reduced.update_pre_compute()
print()
print('small size')
print(planar_lugre_reduced.step(vel))
print()
print('big size')
shape = surf.PObject(20*properties['grid_size'], properties['grid_shape'], surf.p_square)
planar_lugre_reduced.update_p_x_y(shape)
print(planar_lugre_reduced.step(vel))
planar_lugre_reduced.update_pre_compute()
print()
print('update')
print(planar_lugre_reduced.step(vel))

print()
print('re init')
planar_lugre_reduced = ReducedFrictionModel(properties=properties)
planar_lugre_reduced.update_p_x_y(shape)
planar_lugre_reduced.update_pre_compute()
print(planar_lugre_reduced.step(vel))














