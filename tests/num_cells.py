"""
This file test how fast the distributed model converges with increased resolution of the contact surface.
"""
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import frictionModelsCPP.build.FrictionModelCPPClass as cpp
from tests.velocity_profiles import vel_num_cells
import matplotlib as mpl
import surfaces.surfaces as surf
from frictionModels.utils import vel_to_cop

shape_set = {'Square': surf.p_square, 'Circle': surf.p_circle, 'Line': surf.p_line, 'LineGrad': surf.p_line_grad}

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

plot_mesh_size = 1
n_skipp = 10
sim_time = 5
dt = 1e-5
fn = 1
n_baseline = 101
num_time_steps = int(sim_time/dt)
time = []
data_vel = {'x':[], 'y':[], 'tau':[]}
data_base_line = {}
data = {}
contact_size = 0.02

def properties_to_list(prop):
    list_ = []
    for index, key in enumerate(prop):
        if key == "grid_shape":
            list_.append(prop[key][0])
            list_.append(prop[key][1])
        else:
            list_.append(prop[key])
    return list_


shapes = ['Square', 'Circle', 'LineGrad']
n_grids = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]

f_params = [{'mu_c':1, 'mu_s':1, 's2':0.0}, {'mu_c':1, 'mu_s':1.2, 's2':0.2}]

for i_param, f_param in enumerate(f_params):
    for i_shape, shape in enumerate(shapes):
        p = {'grid_shape': (n_baseline, n_baseline),  # number of grid elements in x any
             'grid_size': contact_size / n_baseline,  # the physical size of each grid element
             'mu_c': f_param['mu_c'],
             'mu_s': f_param['mu_s'],
             'v_s': 1e-3,
             'alpha': 2, # called gamma in paper
             's0': 1e6,
             's1': 8e2,
             's2': f_param['s2'],
             'dt': dt,
             'z_ba_ratio': 0.9,
             'stability': False,
             'elasto_plastic': True,
             'steady_state': False}

        fic = cpp.DistributedFrictionModel()
        fic.init(properties_to_list(p), shape, fn)
        cop = np.array(fic.get_cop())

        data_temp = {'x': [], 'y': [], 'tau': []}

        for i_t in tqdm(range(num_time_steps)):
            t = i_t * dt
            vel = vel_num_cells(t)
            vel = vel_to_cop(-cop, vel_vec=vel)

            vel_cpp = [vel['x'], vel['y'], vel['tau']]
            a = fic.step(vel_cpp)
            a = fic.get_force_at_cop()

            if i_t % n_skipp == 0:
                data_temp['x'].append(a[0])
                data_temp['y'].append(a[1])
                data_temp['tau'].append(a[2])
        data_base_line[shape + "_"+str(i_param)] = data_temp

        for i_n_grid, n_grid in enumerate(n_grids):

            p = {'grid_shape': (n_grid, n_grid),  # number of grid elements in x any
                 'grid_size': contact_size/n_grid,  # the physical size of each grid element
                 'mu_c': f_param['mu_c'],
                 'mu_s': f_param['mu_s'],
                 'v_s': 1e-3,
                 'alpha': 2,
                 's0': 1e6,
                 's1': 8e2,
                 's2': f_param['s2'],
                 'dt': dt,
                 'z_ba_ratio': 0.9,
                 'stability': False,
                 'elasto_plastic': True,
                 'steady_state': False}

            fic = cpp.DistributedFrictionModel()
            fic.init(properties_to_list(p), shape, fn)

            data_temp = {'x':[], 'y':[], 'tau':[]}

            for i_t in tqdm(range(num_time_steps)):
                t = i_t*dt
                vel = vel_num_cells(t)
                vel = vel_to_cop(-cop, vel_vec=vel)

                vel_cpp = [vel['x'], vel['y'],vel['tau']]
                a = fic.step(vel_cpp)
                a = fic.get_force_at_cop()

                if i_t%n_skipp == 0:
                    if i_param == 0 and i_shape == 0 and i_n_grid == 0:
                        time.append(t)
                        data_vel['x'].append(vel['x'])
                        data_vel['y'].append(vel['y'])
                        data_vel['tau'].append(vel['tau'])

                    data_temp['x'].append(a[0])
                    data_temp['y'].append(a[1])
                    data_temp['tau'].append(a[2])
            data[shape + "_"+str(i_param)+"_"+str(n_grid)] = data_temp


def get_rmse_curve(shape, i_p, n_grids, data, data_base):
    d_base = data_base[shape+"_"+str(i_p)]

    n_d_base_x = np.array(d_base['x'])
    n_d_base_y = np.array(d_base['y'])
    n_d_base_tau = np.array(d_base['tau'])
    xy_max = np.max(abs(np.linalg.norm([n_d_base_x, n_d_base_y], axis=0)))
    tau_max = np.max(abs(n_d_base_tau))

    out_xy = []
    out_tau = []
    for i, v in enumerate(n_grids):
        d = data[shape + "_"+str(i_p)+"_"+str(v)]
        n_d_x = np.array(d['x'])
        n_d_y = np.array(d['y'])
        n_d_tau = np.array(d['tau'])

        rmse_xy = np.sqrt(np.mean((n_d_base_x - n_d_x)**2 + (n_d_base_y - n_d_y)**2))/xy_max
        rmse_tau = np.sqrt(np.mean((n_d_base_tau - n_d_tau)**2))/tau_max
        out_xy.append(rmse_xy)
        out_tau.append(rmse_tau)

    return n_grids, out_xy, out_tau

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 12))
for i_param, f_param in enumerate(f_params):
    print(i_param, len(f_param))
    for i_shape, shape in enumerate(shapes):
        r1, r2, r3 = get_rmse_curve(shape, i_param, n_grids, data, data_base_line)
        if plot_mesh_size:
            r = [contact_size / rx for rx in r1]
        else:
            r = r1
        ax1.plot(r, r2, label=shape+" p="+str(i_param), alpha=0.7)
        ax2.plot(r, r3, label=shape+" p="+str(i_param), alpha=0.7)

ax1.legend(loc=1)
if plot_mesh_size:
    ax1.set_xlabel('Meshsize [m]')
else:
    ax1.set_xlabel('N')
ax1.set_ylabel('rmse$/f_{max}$')
ax1.set_title('Tangential friction force')

ax2.legend(loc=1)
if plot_mesh_size:
    ax2.set_xlabel('Meshsize [m]')
else:
    ax2.set_xlabel('N')
ax2.set_ylabel('rmse$/\\tau_{max}$')
ax2.set_title('Rotational friction force')

ax3.plot(time, data_vel['x'], label='$v_x$')
ax3.plot(time, data_vel['y'], label='$v_y$')
ax3.legend(loc=1)
ax3.set_xlabel('Time $[s]$')
ax3.set_ylabel('Velocity $[m/s]$')
ax3.set_title('Tangential velocity profile')

ax4.plot(time, data_vel['tau'], label='$\omega$')
ax4.legend(loc=1)
ax4.set_title('Angular velocity profile')
ax4.set_xlabel('Time $[s]$')
ax4.set_ylabel('Velocity $[rad/s]$')


plt.tight_layout()
plt.show()


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3))

for i_param, f_param in enumerate(f_params):
    print(i_param, len(f_param))
    for i_shape, shape in enumerate(shapes):
        r1, r2, r3 = get_rmse_curve(shape, i_param, n_grids, data, data_base_line)
        if plot_mesh_size:
            r = [contact_size / rx for rx in r1]
        else:
            r = r1
        ax1.plot(r, r2, label=shape+" p="+str(i_param), alpha=0.7)
        ax2.plot(r, r3, label=shape+" p="+str(i_param), alpha=0.7)

ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('RMSE$/||\mathbf{f}_{t\max}||$', fontsize="10" )
ax1.get_xaxis().set_visible(False)

ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('RMSE$/\\tau_{\max}$', fontsize="10")
if plot_mesh_size:
    ax2.set_xlabel('Meshsize [m]')
else:
    ax2.set_xlabel('N')

plt.tight_layout(h_pad=0.015)
plt.show()



