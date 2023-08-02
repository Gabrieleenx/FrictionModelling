"""
This file compares the distributed and reduced models.
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl
import surfaces.surfaces as surf
from tests.velocity_profiles import *
import frictionModelsCPP.build.FrictionModelCPPClass as cpp
import frictionModelsCPP.build.ReducedFrictionModelCPPClass as red_cpp
from frictionModels.utils import vel_to_cop

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0.01
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

fn = 1.0
r = 0.01
time = 5

properties = {'grid_shape': (21, 21),  # number of grid elements in x any
              'grid_size': 2*r/21,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.2,
              'v_s': 1e-3,
              'alpha': 2, # called gamma in paper
              's0': 1e6,
              's1': 8e1,
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

fic = cpp.DistributedFrictionModel()

shape_name = "LineGrad"

fic.init(properties_to_list(properties), shape_name, fn)
cop = np.array(fic.get_cop())
fic_red = red_cpp.ReducedFrictionModel()
fic_red.init(properties_to_list(properties), shape_name, fn)

n_steps = int(time / properties['dt'])

data_vel = np.zeros((4, n_steps))  # t, vx, vy, v_tau
data_distributed_cpp = np.zeros((4, n_steps))  # t, fx, fy, f_tau
data_reduced_cpp = np.zeros((4, n_steps))  # t, fx, fy, f_tau

# running simulation
for i in tqdm(range(n_steps)):
    t = (i) * properties['dt']
    vel = vel_num_cells(t)

    data_vel[0, i] = t
    data_vel[1, i] = vel['x']
    data_vel[2, i] = vel['y']
    data_vel[3, i] = vel['tau']
    vel = vel_to_cop(-cop, vel_vec=vel)

    f_ = fic.step([vel['x'], vel['y'], vel['tau']])
    f_ = fic.get_force_at_cop()
    f = {'x':f_[0],'y':f_[1],'tau':f_[2]}
    data_distributed_cpp[0, i] = t
    data_distributed_cpp[1, i] = f['x']
    data_distributed_cpp[2, i] = f['y']
    data_distributed_cpp[3, i] = f['tau']

for i in tqdm(range(n_steps)):
    t = (i) * properties['dt']
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

ft_max = np.max(abs(np.linalg.norm([data_distributed_cpp[1, :], data_distributed_cpp[2, :]], axis=0)))
f_tau_max = np.max(abs(data_distributed_cpp[3, :]))
print('RMSE tangential/ft_max', np.sqrt(np.mean((data_distributed_cpp[1, :] - data_reduced_cpp[1, :]) ** 2 + (data_distributed_cpp[2, :] - data_reduced_cpp[2, :]) ** 2)) / ft_max, ' ft_max ', ft_max)
print('RMSE angular/ftau_max', np.sqrt(np.mean((data_distributed_cpp[3, :] - data_reduced_cpp[3, :]) ** 2)) / f_tau_max, ' f_tau_max ', f_tau_max)

# plotting

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 5))
#f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3))
ax1.plot(data_distributed_cpp[0, :], data_distributed_cpp[1, :], alpha=0.7, label='$f_x$')
ax1.plot(data_distributed_cpp[0, :], data_distributed_cpp[2, :], alpha=0.7, label='$f_y$')
ax1.plot(data_reduced_cpp[0, :], data_reduced_cpp[1, :], '--', label='$\\bar{f_x}$')
ax1.plot(data_reduced_cpp[0, :], data_reduced_cpp[2, :], '--', label='$\\bar{f_y}$')
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Force [N]', fontsize="10" )
ax1.get_xaxis().set_visible(False)

ax2.plot(data_distributed_cpp[0, :], data_distributed_cpp[3, :], label='$\\tau$')
ax2.plot(data_reduced_cpp[0, :], data_reduced_cpp[3, :], '--', label='$\\bar{\\tau}$')
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Torque [Nm]', fontsize="10" )
ax2.get_xaxis().set_visible(False)
#ax2.set_xlabel('Time [s]')

ax3.plot(data_vel[0, :], data_vel[1, :], label='$v_x$')
ax3.plot(data_vel[0, :], data_vel[2, :], label='$v_y$')
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax3.get_yaxis().set_label_coords(-0.11,0.5)
ax3.set_ylabel('Velocity [m/s]', fontsize="10" )
ax3.get_xaxis().set_visible(False)

ax4.plot(data_vel[0, :], data_vel[3, :], label='$\omega$')
ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax4.get_yaxis().set_label_coords(-0.11,0.5)
ax4.set_ylabel('Velocity $[rad/s]$', fontsize="10" )
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Velocity [rad/s]')

plt.tight_layout(h_pad=0.015)
plt.show()
