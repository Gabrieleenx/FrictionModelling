import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import surfaces.surfaces as surf
from velocity_profiles import *
from frictionModels.frictionModel import FullFrictionModel, ReducedFrictionModel

properties = {'grid_shape': (20, 20),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.3,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e5,
              's1': 2e1,
              's2': 0.4,
              'dt': 1e-3,
              'stability': True,
              'elasto_plastic': True,
              'z_ba_ratio': 0.9,
              'steady_state': False}



shape = surf.PObject(properties['grid_size'], properties['grid_shape'], surf.p_line)
time = 2

n_steps = int(time / properties['dt'])

# initialize friction models

planar_lugre = FullFrictionModel(properties=properties)
planar_lugre.update_p_x_y(shape)

planar_lugre_reduced = ReducedFrictionModel(properties=properties)
planar_lugre_reduced.update_p_x_y(shape)
planar_lugre_reduced.update_pre_compute()

# initialize data collection

data_vel = np.zeros((4, n_steps))  # t, vx, vy, v_tau
data_full = np.zeros((4, n_steps))  # t, fx, fy, f_tau
data_reduced = np.zeros((4, n_steps))  # t, fx, fy, f_tau

# running simulation

print('Full model')
for i in tqdm(range(n_steps)):
    t = i * properties['dt']
    vel = vel_gen_5(t)
    data_vel[0, i] = t
    data_vel[1, i] = vel['x']
    data_vel[2, i] = vel['y']
    data_vel[3, i] = vel['tau']

    f = planar_lugre.step(vel_vec=vel)
    data_full[0, i] = t
    data_full[1, i] = f['x']
    data_full[2, i] = f['y']
    data_full[3, i] = f['tau']

print('Reduced model')

for i in tqdm(range(n_steps)):
    t = i * properties['dt']
    vel = vel_gen_5(t)
    data_vel[0, i] = t
    data_vel[1, i] = vel['x']
    data_vel[2, i] = vel['y']
    data_vel[3, i] = vel['tau']

    f = planar_lugre_reduced.step(vel_vec=vel)
    data_reduced[0, i] = t
    data_reduced[1, i] = f['x']
    data_reduced[2, i] = f['y']
    data_reduced[3, i] = f['tau']

# plotting

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))

ax1.plot(data_full[0, :], data_full[1, :], alpha=0.7, label='fx')
ax1.plot(data_full[0, :], data_full[2, :], alpha=0.7, label='fy')
ax1.plot(data_reduced[0, :], data_reduced[1, :], '--', label='fx red')
ax1.plot(data_reduced[0, :], data_reduced[2, :], '--', label='fy red')
ax1.set_title('Force')
ax1.set_xlabel('Time')
ax1.set_ylabel('Force [N]')
ax1.legend()

ax2.plot(data_full[0, :], data_full[3, :], label='f tau')
ax2.plot(data_reduced[0, :], data_reduced[3, :], '--', label='f tau red')
ax2.set_title('Torque')
ax2.set_xlabel('Time')
ax2.set_ylabel('Torque [Nm]')
ax2.legend()

ax3.plot(data_vel[0, :], data_vel[1, :], label='vx')
ax3.plot(data_vel[0, :], data_vel[2, :], label='vy')
ax3.set_title('Velocity profile')
ax3.set_xlabel('Time')
ax3.set_ylabel('Velocity [m/s]')
ax3.legend()

ax4.plot(data_vel[0, :], data_vel[3, :], label='vTau')
ax4.set_title('Angular velocity profile')
ax4.set_xlabel('Time')
ax4.set_ylabel('Velocity [rad/s]')
ax4.legend()

plt.tight_layout()
plt.show()
