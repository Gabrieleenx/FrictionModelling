"""
This file compares the distributed and reduced models.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl
from tests.velocity_profiles import *
from friction_models_cpp import ModelRed, ModelDist, Force, ModelEllipse
from frictionModels.utils import vel_to_cop
from surfaces.surfaces import non_convex_1, non_convex_2, p_line_grad
from scipy.integrate import ode, solve_ivp
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0.01
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

fn = 1.0
time = 5
r = 0.01
cells = 21
atol = 1e-8
rtol = 1e-6
shape = "NonConvex1"
properties = {'grid_shape': (cells, cells),  # number of grid elements in x any
              'grid_size': 2*r/cells,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.2,
              'v_s': 1e-3,
              'alpha': 2,  # called gamma in paper
              's0': 1e6,
              's1': 8e2,
              's2': 0.2,
              'dt': 0.01,
              'z_ba_ratio': 0.9,
              'stability': False,
              'elasto_plastic': True,
              'steady_state': False,
              'n_ls': 20}


n_steps = int(time / properties['dt'])

data_vel = np.zeros((4, n_steps))  # t, vx, vy, v_tau
data_distributed_cpp = np.zeros((4, n_steps))  # t, fx, fy, f_tau
data_reduced_cpp = np.zeros((4, n_steps))  # t, fx, fy, f_tau


model_red = ModelRed(shape, properties)
#model_red = ModelEllipse(shape, properties)
model = ModelDist(shape, properties)

if shape == 'NonConvex1':
    p_surf = non_convex_1(properties["grid_shape"], 1)
    model.update_surface(p_surf, "non_convex")
    model_red.update_surface(p_surf, "non_convex")

if shape == 'NonConvex2':
    p_surf = non_convex_2(properties["grid_shape"], 1)
    model.update_surface(p_surf, "non_convex")
    model_red.update_surface(p_surf, "non_convex")


cop = np.array(model.model_fric.get_cop())
cop_red = np.array(model_red.model_fric.get_cop())
print(cop)
y = model.y_init
y_red = model_red.y_init

force = Force()
force_red = Force()

dt = properties['dt']
# running simulation

for i in tqdm(range(n_steps)):
    t = (i) * properties['dt']
    vel = vel_num_cells(t)

    data_vel[0, i] = t
    data_vel[1, i] = vel['x']
    data_vel[2, i] = vel['y']
    data_vel[3, i] = vel['tau']
    vel_red = vel_to_cop(-cop_red, vel_vec=vel)
    vel = vel_to_cop(-cop, vel_vec=vel)

    vel_ = [vel['x'], vel['y'], vel['tau']]
    vel_red_ = [vel_red['x'], vel_red['y'], vel_red['tau']]

    sol = solve_ivp(model.f, (0, dt), y, method='LSODA', t_eval=[dt], args=[vel_, force], atol=atol, rtol=rtol, max_step=1e-3)
    y = sol.y[:, -1]  # Update initial conditions for the next step
    _ = model.f(t, y, vel_, force)  # to get the correct force

    data_distributed_cpp[0, i] = t
    data_distributed_cpp[1, i] = force.fx_cop
    data_distributed_cpp[2, i] = force.fy_cop
    data_distributed_cpp[3, i] = force.tau_cop


for i in tqdm(range(n_steps)):
    t = (i) * properties['dt']
    vel = vel_num_cells(t)
    data_vel[0, i] = t
    data_vel[1, i] = vel['x']
    data_vel[2, i] = vel['y']
    data_vel[3, i] = vel['tau']
    vel_red = vel_to_cop(-cop_red, vel_vec=vel)
    vel = vel_to_cop(-cop, vel_vec=vel)

    vel_ = [vel['x'], vel['y'], vel['tau']]
    vel_red_ = [vel_red['x'], vel_red['y'], vel_red['tau']]

    sol_red = solve_ivp(model_red.f, (0, dt), y_red, method='LSODA', t_eval=[dt], args=[vel_red_, force_red], atol=atol, rtol=rtol, max_step=1e-3)
    y_red = sol_red.y[:, -1]  # Update initial conditions for the next step
    _ = model_red.f(t, y_red, vel_red_, force_red)  # to get the correct force

    data_reduced_cpp[0, i] = t
    data_reduced_cpp[1, i] = force_red.fx_cop
    data_reduced_cpp[2, i] = force_red.fy_cop
    data_reduced_cpp[3, i] = force_red.tau_cop

# plotting

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 5))
#f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3))
ax1.plot(data_distributed_cpp[0, :], data_distributed_cpp[1, :], alpha=0.7, label='$f_{d,x}$')
ax1.plot(data_distributed_cpp[0, :], data_distributed_cpp[2, :], alpha=0.7, label='$f_{d,y}$')
ax1.plot(data_reduced_cpp[0, :], data_reduced_cpp[1, :], '--', label='$f_{r,x}$')
ax1.plot(data_reduced_cpp[0, :], data_reduced_cpp[2, :], '--', label='$f_{r,y}$')
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Force [N]', fontsize="10" )
ax1.get_xaxis().set_visible(False)

ax2.plot(data_distributed_cpp[0, :], data_distributed_cpp[3, :], label='$\\tau_d$')
#ax2.plot(data_reduced_cpp[0, :], data_reduced_cpp[3, :], '--', label='$\\tau_r$')
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
# plotting

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3))
ax1.plot(data_distributed_cpp[0, :], data_distributed_cpp[1, :], alpha=0.7, label='$f_{d,x}$')
ax1.plot(data_distributed_cpp[0, :], data_distributed_cpp[2, :], alpha=0.7, label='$f_{d,y}$')
ax1.plot(data_reduced_cpp[0, :], data_reduced_cpp[1, :], '--', label='$f_{r,x}$')
ax1.plot(data_reduced_cpp[0, :], data_reduced_cpp[2, :], '--', label='$f_{r,y}$')
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Force [N]', fontsize="10" )
ax1.get_xaxis().set_visible(False)

ax2.plot(data_distributed_cpp[0, :], data_distributed_cpp[3, :], label='$\\tau_d$')
ax2.plot(data_reduced_cpp[0, :], data_reduced_cpp[3, :], '--', label='$\\tau_r$')
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Torque [Nm]', fontsize="10" )
#ax2.get_xaxis().set_visible(False)
ax2.set_xlabel('Time [s]')

plt.tight_layout(h_pad=0.015)
plt.show()
