"""
This file test the change of contact size for the reduced model without the need to recalculate the limit surface.
"""
import surfaces.surfaces as surf
import matplotlib as mpl
from frictionModels.frictionModel import ReducedFrictionModel

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']

properties = {'grid_shape': (20, 20),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.3,
              'v_s': 1e-3,
              'alpha': 2, # called gamma in paper
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
print('Small contact size')
print(planar_lugre_reduced.step(vel))
print()
print('Change to larger contact size')
shape = surf.PObject(20*properties['grid_size'], properties['grid_shape'], surf.p_square)
planar_lugre_reduced.update_p_x_y(shape)
print(planar_lugre_reduced.step(vel))
print()
print('Update the pre-computation')
planar_lugre_reduced.update_pre_compute()
print(planar_lugre_reduced.step(vel))

print()
print('Reinitialize')
planar_lugre_reduced = ReducedFrictionModel(properties=properties)
planar_lugre_reduced.update_p_x_y(shape)
planar_lugre_reduced.update_pre_compute()
print(planar_lugre_reduced.step(vel))














