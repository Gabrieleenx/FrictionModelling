import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
import surfaces.surfaces as surf
from tests.velocity_profiles import *
from frictionModels.frictionModel import FullFrictionModel, ReducedFrictionModel
import frictionModelsCPP.build.FrictionModelCPPClass as cpp
import frictionModelsCPP.build.ReducedFrictionModelCPPClass as red_cpp
from frictionModels.utils import vel_to_cop

properties = {'grid_shape': (21, 21),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.2,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e5,
              's1': 2e1,
              's2': 0.2,
              'dt': 1e-4,
              'z_ba_ratio': 0.9,
              'stability': True,
              'elasto_plastic': True,
              'steady_state': False,
              'n_ls': 20}
def properties_to_list(prop):
    list_ = []
    for index, key in enumerate(prop):
        if key == "grid_shape":
            list_.append(prop[key][0])
            list_.append(prop[key][1])
        else:
            list_.append(prop[key])
    return list_

fic = cpp.FullFrictionModel()

shape_name = "LineGrad"
fn = 1.0
fic.init(properties_to_list(properties), shape_name, fn)

fic_red = red_cpp.ReducedFrictionModel()
fic_red.init(properties_to_list(properties), shape_name, fn)

shape = surf.PObject(properties['grid_size'], properties['grid_shape'], surf.p_line_grad)
time = 5

n_steps = int(time / properties['dt'])

# initialize friction models

planar_lugre = FullFrictionModel(properties=properties)
planar_lugre.update_p_x_y(shape)
cop = planar_lugre.cop
planar_lugre_reduced = ReducedFrictionModel(properties=properties, nr_ls_segments=20)
planar_lugre_reduced.update_p_x_y(shape)
planar_lugre_reduced.update_pre_compute()

# initialize data collection

data_vel = np.zeros((4, n_steps))  # t, vx, vy, v_tau
data_full = np.zeros((4, n_steps))  # t, fx, fy, f_tau
data_full_cpp = np.zeros((4, n_steps))  # t, fx, fy, f_tau
data_reduced = np.zeros((4, n_steps))  # t, fx, fy, f_tau
data_reduced_cpp = np.zeros((4, n_steps))  # t, fx, fy, f_tau
"""
# running simulation
print('Full model')
for i in tqdm(range(n_steps)):
    t = (i) * properties['dt']
    #vel = vel_gen_5(t)
    vel = vel_num_cells(t)
    data_vel[0, i] = t
    data_vel[1, i] = vel['x']
    data_vel[2, i] = vel['y']
    data_vel[3, i] = vel['tau']


    f = planar_lugre.step(vel_vec=vel)
    #f_ = fic.step([vel['x'], vel['y'], vel['tau']])
    #f = {'x':f_[0],'y':f_[1],'tau':f_[2]}
    data_full[0, i] = t
    data_full[1, i] = f['x']
    data_full[2, i] = f['y']
    data_full[3, i] = f['tau']
"""

# running simulation
print('Full model data_full_cpp')
for i in tqdm(range(n_steps)):
    t = (i) * properties['dt']
    # vel = vel_gen_5(t)
    vel = vel_num_cells(t)

    data_vel[0, i] = t
    data_vel[1, i] = vel['x']
    data_vel[2, i] = vel['y']
    data_vel[3, i] = vel['tau']
    vel = vel_to_cop(-cop, vel_vec=vel)

    f_ = fic.step([vel['x'], vel['y'], vel['tau']])
    f_ = fic.get_force_at_cop()
    f = {'x':f_[0],'y':f_[1],'tau':f_[2]}
    data_full_cpp[0, i] = t
    data_full_cpp[1, i] = f['x']
    data_full_cpp[2, i] = f['y']
    data_full_cpp[3, i] = f['tau']


print('Reduced model')

for i in tqdm(range(n_steps)):
    t = i * properties['dt']
    # vel = vel_gen_5(t)
    vel = vel_num_cells(t)
    data_vel[0, i] = t
    data_vel[1, i] = vel['x']
    data_vel[2, i] = vel['y']
    data_vel[3, i] = vel['tau']
    vel = vel_to_cop(-cop, vel_vec=vel)

    f = planar_lugre_reduced.step(vel_vec=vel)
    f = planar_lugre_reduced.force_at_cop
    data_reduced[0, i] = t
    data_reduced[1, i] = f['x']
    data_reduced[2, i] = f['y']
    data_reduced[3, i] = f['tau']

print('Reduced model cpp')

for i in tqdm(range(n_steps)):
    t = (i) * properties['dt']
    # vel = vel_gen_5(t)
    vel = vel_num_cells(t)

    data_vel[0, i] = t
    data_vel[1, i] = vel['x']
    data_vel[2, i] = vel['y']
    data_vel[3, i] = vel['tau']
    vel = vel_to_cop(-cop, vel_vec=vel)

    f_ = fic_red.step([vel['x'], vel['y'], vel['tau']])
    f_ = fic_red.get_force_at_cop()
    f = {'x':f_[0],'y':f_[1],'tau':f_[2]}
    data_reduced_cpp[0, i] = t
    data_reduced_cpp[1, i] = f['x']
    data_reduced_cpp[2, i] = f['y']
    data_reduced_cpp[3, i] = f['tau']





ft_max = np.max(abs(np.linalg.norm([data_full[1,:], data_full[2,:]], axis=0)))
f_tau_max = np.max(abs(data_full[3,:]))
print('RMSE tangential/ft_max cpp', np.sqrt(np.mean((data_full[1,:] - data_reduced_cpp[1, :])**2 + (data_full[2,:] - data_reduced_cpp[2, :])**2))/ft_max, ft_max)
print('RMSE angular/ftau_max cpp', np.sqrt(np.mean((data_full[3,:] - data_reduced_cpp[3, :])**2))/f_tau_max, f_tau_max)
print('RMSE tangential/ft_max ', np.sqrt(np.mean((data_full[1,:] - data_reduced[1, :])**2 + (data_full[2,:] - data_reduced[2, :])**2))/ft_max, ft_max)
print('RMSE angular/ftau_max ', np.sqrt(np.mean((data_full[3,:] - data_reduced[3, :])**2))/f_tau_max, f_tau_max)


# plotting
sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 2})

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 12))

ax1.plot(data_full_cpp[0, :], data_full_cpp[1, :], alpha=0.7, label='$f_x$')
ax1.plot(data_full_cpp[0, :], data_full_cpp[2, :], alpha=0.7, label='$f_y$')
#ax1.plot(data_full[0, :], data_full[1, :], alpha=0.7, label='fx')
#ax1.plot(data_full[0, :], data_full[2, :], alpha=0.7, label='fy')
ax1.plot(data_reduced[0, :], data_reduced[1, :], '--', label='fx red')
ax1.plot(data_reduced[0, :], data_reduced[2, :], '--', label='fy red')
ax1.plot(data_reduced_cpp[0, :], data_reduced_cpp[1, :], '--', label='$f_x$ reduced')
ax1.plot(data_reduced_cpp[0, :], data_reduced_cpp[2, :], '--', label='$f_y$ reduced')

ax1.set_title('Tangential friction force')
ax1.set_xlabel('Time $[s]$')
ax1.set_ylabel('Force $[N]$')
ax1.legend()

ax2.plot(data_full_cpp[0, :], data_full_cpp[3, :], label='$\\tau$')
#ax2.plot(data_full[0, :], data_full[3, :], label='f tau')
ax2.plot(data_reduced[0, :], data_reduced[3, :], '--', label='f tau red')
ax2.plot(data_reduced_cpp[0, :], data_reduced_cpp[3, :], '--', label='$\\tau$ reduced')

ax2.set_title('Rotational friction force')
ax2.set_xlabel('Time $[s]$')
ax2.set_ylabel('Torque $[Nm]$')
ax2.legend()

ax3.plot(data_vel[0, :], data_vel[1, :], label='$v_x$')
ax3.plot(data_vel[0, :], data_vel[2, :], label='$v_y$')
ax3.set_xlabel('Time $[s]$')
ax3.set_ylabel('Velocity $[m/s]$')
ax3.set_title('Tangential velocity profile')
ax3.legend()

ax4.plot(data_vel[0, :], data_vel[3, :], label='$\omega$')
ax4.set_title('Angular velocity profile')
ax4.set_xlabel('Time $[s]$')
ax4.set_ylabel('Velocity $[rad/s]$')
ax4.legend()

plt.tight_layout()
plt.show()
