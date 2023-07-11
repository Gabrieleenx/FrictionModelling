import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0.01
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

import surfaces.surfaces as surf
from tests.velocity_profiles import *
from frictionModels.frictionModel import FullFrictionModel, ReducedFrictionModel
import frictionModelsCPP.build.FrictionModelCPPClass as cpp
import frictionModelsCPP.build.ReducedFrictionModelCPPClass as red_cpp
from frictionModels.utils import vel_to_cop
r = 0.01

properties = {'grid_shape': (21, 21),  # number of grid elements in x any
              'grid_size': 2*r/21,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e6,
              's1': 8e1,
              's2': 0,
              'dt': 1e-4,
              'z_ba_ratio': 0.9,
              'stability': True,
              'elasto_plastic': True,
              'steady_state': True,
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

shape_name = "Line"
fn = 1.0

shape = surf.PObject(properties['grid_size'], properties['grid_shape'], surf.p_line)
time = 5
n_steps = int(time / properties['dt'])

# initialize friction models

planar_lugre = FullFrictionModel(properties=properties)
planar_lugre.update_p_x_y(shape)

cop = planar_lugre.cop

# initialize data collection

data_vel = np.zeros((4, n_steps))  # t, vx, vy, v_tau
data_full = np.zeros((4, n_steps))  # t, fx, fy, f_tau
data_full_bilinear = np.zeros((4, n_steps))  # t, fx, fy, f_tau


# running simulation
print('Full model')
for i in tqdm(range(n_steps)):
    t = (i) * properties['dt']
    vel = vel_num_cells(t)
    data_vel[0, i] = t
    data_vel[1, i] = vel['x']
    data_vel[2, i] = vel['y']
    data_vel[3, i] = vel['tau']
    vel = vel_to_cop(-cop, vel_vec=vel)

    f = planar_lugre.step_single_point(vel_vec=vel)

    data_full[0, i] = t
    data_full[1, i] = f['x']
    data_full[2, i] = f['y']
    data_full[3, i] = f['tau']

    f = planar_lugre.step(vel_vec=vel)

    data_full_bilinear[0, i] = t
    data_full_bilinear[1, i] = f['x']
    data_full_bilinear[2, i] = f['y']
    data_full_bilinear[3, i] = f['tau']

# plotting

#f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 5))
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3))
ax1.plot(data_full[0, :], data_full[1, :], alpha=0.7, label='$f_x$')
ax1.plot(data_full[0, :], data_full[2, :], alpha=0.7, label='$f_y$')
ax1.plot(data_full_bilinear[0, :], data_full_bilinear[1, :], '--', alpha=1, label='$f_x$ b-l')
ax1.plot(data_full_bilinear[0, :], data_full_bilinear[2, :], '--', alpha=1, label='$f_y$ b-l')
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Force [N]', fontsize="10" )
ax1.get_xaxis().set_visible(False)

# Make the zoom-in plot:

axins = zoomed_inset_axes(ax1, 4, loc=3) # zoom = 2
axins.plot(data_full[0, :], data_full[1, :], alpha=0.7, label='$f_x$')
axins.plot(data_full_bilinear[0, :], data_full_bilinear[1, :], '--', alpha=1, label='$f_x$ b-l')

axins.set_xlim(1.85, 2.05)
axins.set_ylim(-0.7, -0.5)
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelright=False,
    labelleft=False,
    labeltop=False)
mark_inset(ax1, axins, loc1=1, loc2=4, fc="none", ec="0.5")


ax2.plot(data_full[0, :], data_full[3, :], label='$\\tau$')
ax2.plot(data_full_bilinear[0, :], data_full_bilinear[3, :], '--' ,label='$\\tau$ b-l')

ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Torque [Nm]', fontsize="10" )

axins = zoomed_inset_axes(ax2, 4, loc=2) # zoom = 2
axins.plot(data_full[0, :], data_full[3, :], label='$\\tau$')
axins.plot(data_full_bilinear[0, :], data_full_bilinear[3, :], '--' ,label='$\\tau$ b-l')

axins.set_xlim(1.85, 2.05)
axins.set_ylim(-0.003, -0.002)
#axins.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.10f'))

axins.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
    labelright=False)

mark_inset(ax2, axins, loc1=1, loc2=4, fc="none", ec="0.5")
#ax2.get_xaxis().set_visible(False)

ax2.set_xlabel('Time [s]')

"""
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
"""
plt.tight_layout(h_pad=0.015)
plt.show()
