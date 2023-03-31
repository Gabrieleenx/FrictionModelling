from utils import vel_to_cop, CustomHashList3D
from surfaces.surfaces import p_square, p_line, p_circle, p_line_grad
from frictionModel import FullFrictionModel, ReducedFrictionModel
import numpy as np

properties = {'grid_shape': (21, 21),  # number of grid elements in x any
              'grid_size': 0.001,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1,
              'v_s': 0.01,
              'alpha': 2,
              's0': 1e5,
              's1': 20,
              's2': 0,
              'dt': 1e-4,
              'stability': True,
              'elasto_plastic': True,
              'z_ba_ratio': 0.9,
              'steady_state': True}
shape_set = {'Square': p_square, 'Circle': p_circle, 'Line': p_line, 'LineGrad': p_line_grad}
shape = 'LineGrad'
planar_lugre = FullFrictionModel(properties=properties)
planar_lugre_reduced = ReducedFrictionModel(properties=properties, nr_ls_segments=20)

planar_lugre.update_p_x_y(shape_set[shape])
planar_lugre_reduced.update_p_x_y(shape_set[shape])
planar_lugre_reduced.update_pre_compute()
vel = {'x': 0.01, 'y': 0.01, 'tau': 1}

#vel = {'x': 0.003, 'y': 0.01, 'tau': 1}

vel = {'x': 0.004805664738987199, 'y': 0.012609112885666523, 'tau': 1.105263157894737}
vel = {'x': 0.008974359, 'y': 0.0, 'tau': 2.769231}
vel = {'x': 0.0, 'y': 0.0, 'tau': 1}

cop = planar_lugre.cop
print('vel cop, ', vel_to_cop(cop, vel))
vel = vel_to_cop(-cop, vel)
f_full = planar_lugre.step(vel_vec=vel)
print('f_full', planar_lugre.force_at_cop)
f_ = planar_lugre_reduced.step(vel_vec=vel)
print('f_red', planar_lugre_reduced.force_at_cop)
gamma = planar_lugre_reduced.gamma
ls = planar_lugre_reduced.limit_surface.get_bilinear_interpolation( vel_to_cop(cop, vel), gamma)

ls_ = ls*np.array([0.21, 0.21, 0.0008372589830271525])
print('ls', ls_)

