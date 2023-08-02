"""
This file simulates in-hand stick and slip motion by varying the grip force.
"""
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import frictionModelsCPP.build.FrictionModelCPPClass as cpp
import frictionModelsCPP.build.ReducedFrictionModelCPPClass as red_cpp
from frictionModels.utils import vel_to_cop
from matplotlib.patches import Rectangle, Circle

import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

dt = 1e-4
r = 0.01
sim_time = 2
n = 5
time_stamps = np.arange(n)/(n-1) * (sim_time)
shape = 'Circle'
title_ = 'Circular contact'
p = {'grid_shape': (21, 21),  # number of grid elements in x any
     'grid_size': 2*r / 21,  # the physical size of each grid element
     'mu_c': 1.0,
     'mu_s': 1.2,
     'v_s': 1e-3,
     'alpha': 2, # called gamma in paper
     's0': 1e6,
     's1': 8e1,
     's2': 0.2,
     'dt': dt,
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


def move_force_to_point(force_at_cop, pos):
    f_t = np.array([force_at_cop[0], force_at_cop[1]])
    m = np.cross(-pos, f_t)
    return [force_at_cop[0], force_at_cop[1], force_at_cop[2] + m]

class object_dynamics(object):
    def __init__(self, h, b, m, dt):
        self.dt = dt
        self.m = m
        self.h = h
        self.b = b
        self.I = (m/12) * (b**2 + h**2)
        self.vel = np.zeros(3)
        self.pos = np.array([0.025,0.0,0.0])

    def step(self, f_f):
        f_g = -9.81 * self.m
        fx = f_f[0]
        fy = f_f[1] + f_g
        ftau = f_f[2]

        ax = fx/self.m
        ay = fy/self.m
        atau = ftau/self.I

        self.pos[0] += self.vel[0] * self.dt + (1 / 2) * ax * self.dt ** 2
        self.pos[1] += self.vel[1] * self.dt + (1 / 2) * ay * self.dt ** 2
        self.pos[2] += self.vel[2] * self.dt + (1 / 2) * atau * self.dt ** 2

        self.vel[0] += ax*self.dt
        self.vel[1] += ay*self.dt
        self.vel[2] += atau*self.dt

        return self.vel, self.pos

object_width = 0.15
object_height = 0.08

object_ = object_dynamics(0.15, 0.08, 0.2, dt)
object_red = object_dynamics(0.15, 0.08, 0.2, dt)
object_fn2 = object_dynamics(0.15, 0.08, 0.2, dt)
object_red_fn2 = object_dynamics(0.15, 0.08, 0.2, dt)

fic = cpp.DistributedFrictionModel()
fic.init(properties_to_list(p), shape, 2)

fic_red = red_cpp.ReducedFrictionModel()
fic_red.init(properties_to_list(p), shape, 2)

fic_fn2 = cpp.DistributedFrictionModel()
fic_fn2.init(properties_to_list(p), shape, 2)

fic_red_fn2 = red_cpp.ReducedFrictionModel()
fic_red_fn2.init(properties_to_list(p), shape, 2)

num_time_steps = int(sim_time/dt)
f_f = [0,0,0]
f_f_red = [0,0,0]

f_f2 = [0,0,0]
f_f_red2 = [0,0,0]

data = {'p_x':[], 'p_y':[], 'theta':[],
        'p_x2':[], 'p_y2':[], 'theta2':[],
        'p_x_red':[], 'p_y_red':[], 'theta_red':[],
        'p_x_red2':[], 'p_y_red2':[], 'theta_red2':[],
        't':[], 'fn':[],'fn2':[], 'f_x':[], 'f_y':[], 'f_tau':[]}

i_c = int(0.001/dt)
for i_t in tqdm(range(num_time_steps)):

    fn = 3.6+0.9*np.sin(i_t*dt*2*np.pi*2) - 0.0*i_t*dt
    fn2 = 4 # do a second force profile that gives more rotation
    if 0.6*(num_time_steps/4) < i_t and i_t < (num_time_steps/4):
        fn2=2.9

    vel, pos = object_.step(np.array(f_f))
    vel_ = vel_to_cop(-pos[0:2], {'x':vel[0], 'y':vel[1], 'tau':vel[2]})
    fic.set_fn(fn)
    f = fic.step([vel_['x'], vel_['y'], vel_['tau']])
    f = np.array(f)*2
    f.tolist()
    f_f = move_force_to_point(f, pos[0:2])

    vel_red, pos_red = object_red.step(np.array(f_f_red))
    vel_red_ = vel_to_cop(-pos_red[0:2], {'x': vel_red[0], 'y': vel_red[1], 'tau': vel_red[2]})
    fic_red.set_fn(fn)
    f_red = fic_red.step([vel_red_['x'], vel_red_['y'], vel_red_['tau']])
    f_red = np.array(f_red) * 2
    f_red.tolist()
    f_f_red = move_force_to_point(f_red, pos_red[0:2])

    vel2, pos2 = object_fn2.step(np.array(f_f2))
    vel_2 = vel_to_cop(-pos[0:2], {'x':vel2[0], 'y':vel2[1], 'tau':vel2[2]})
    fic_fn2.set_fn(fn2)
    f2 = fic_fn2.step([vel_2['x'], vel_2['y'], vel_2['tau']])
    f2 = np.array(f2)*2
    f2.tolist()
    f_f2 = move_force_to_point(f2, pos2[0:2])

    vel_red2, pos_red2 = object_red_fn2.step(np.array(f_f_red2))
    vel_red_2 = vel_to_cop(-pos_red2[0:2], {'x': vel_red2[0], 'y': vel_red2[1], 'tau': vel_red2[2]})
    fic_red_fn2.set_fn(fn2)
    f_red2 = fic_red_fn2.step([vel_red_2['x'], vel_red_2['y'], vel_red_2['tau']])
    f_red2 = np.array(f_red2) * 2
    f_red2.tolist()
    f_f_red2 = move_force_to_point(f_red2, pos_red2[0:2])

    if i_t%i_c == 0:
        data['p_x'].append(pos[0])
        data['p_y'].append(pos[1])
        data['theta'].append(pos[2])

        data['p_x_red'].append(pos_red[0])
        data['p_y_red'].append(pos_red[1])
        data['theta_red'].append(pos_red[2])

        data['p_x2'].append(pos2[0])
        data['p_y2'].append(pos2[1])
        data['theta2'].append(pos2[2])

        data['p_x_red2'].append(pos_red2[0])
        data['p_y_red2'].append(pos_red2[1])
        data['theta_red2'].append(pos_red2[2])

        data['f_x'].append(f[0])
        data['f_y'].append(f[1])
        data['f_tau'].append(f[2])
        data['t'].append(i_t*dt)
        data['fn'].append(fn)
        data['fn2'].append(fn2)

def plot_box(x, y, theta, h, w, ax, color, label):
    rotation = 180*theta/np.pi
    rect = Rectangle((x - w / 2, y - h / 2), w, h, angle=rotation, facecolor=color, edgecolor=color, fill=False, alpha=0.7, rotation_point='center', label=label)
    ax.add_patch(rect)
    rect.set_linewidth(2)


f, ax = plt.subplots(1, 1, figsize=(3,2.6))
lim_scale = 0.8
ax.set_xlim(-0.5*object_width, lim_scale*object_width)
ax.set_ylim(-1.4*object_height, 0.9*object_height)
i_t_max = int(sim_time/(dt* i_c))
for t in time_stamps:
    if t>=sim_time:
        t=sim_time-dt
    i_t = int(t/(dt* i_c))

    x = data['p_x'][i_t]
    y = data['p_y'][i_t]
    theta = data['theta'][i_t]
    c = [i_t / i_t_max, (i_t_max - i_t) / i_t_max, (i_t_max - i_t) / i_t_max]  # R,G,B
    plot_box(x,y,theta, object_height, object_width, ax, c, 't = ' + str(np.round(t,2)))
circle = Circle((0,0), r, edgecolor="black", label='Gripper')
circle.set_facecolor((160/256, 138/256, 119/256, 0.5))
ax.add_patch(circle)
circle.set_linewidth(2)
circle2 = Circle((0,0), 0.001, edgecolor="black")
ax.add_patch(circle2)

ax.plot(data['p_x'], data['p_y'], alpha=0.7, label='CoM')
ax.set_xlabel('x [m]', fontsize="10")
ax.set_ylabel('Position y [m]', fontsize="10")
#ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")

plt.tight_layout(h_pad=0.015)
plt.show()

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6,5))
#sns.despine(f)

ax1.plot(data['t'], data['p_x'], label="$x_1$", alpha=0.7)
ax1.plot(data['t'], data['p_x_red'], label="$\\bar{x}_1$", alpha=0.7)
ax1.plot(data['t'], data['p_x2'], '--', label="$x_2$", alpha=0.7)
ax1.plot(data['t'], data['p_x_red2'], '--', label="$\\bar{x}_2$", alpha=0.7)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Position [m]', fontsize="10" )
ax1.get_xaxis().set_visible(False)
#ax1.set_title(title_)
ax2.plot(data['t'], data['p_y'], label="$y_1$", alpha=0.7)
ax2.plot(data['t'], data['p_y_red'], label="$\\bar{y}_1$", alpha=0.7)
ax2.plot(data['t'], data['p_y2'], '--', label="$y_2$", alpha=0.7)
ax2.plot(data['t'], data['p_y_red2'], '--', label="$\\bar{y}_2$", alpha=0.7)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Position [m]', fontsize="10" )
ax2.get_xaxis().set_visible(False)

ax3.plot(data['t'], data['theta'], label="$\\theta_1$", alpha=0.7)
ax3.plot(data['t'], data['theta_red'], label="$\\bar{\\theta}_1$", alpha=0.7)
ax3.plot(data['t'], data['theta2'], '--', label="$\\theta_2$", alpha=0.7)
ax3.plot(data['t'], data['theta_red2'], '--', label="$\\bar{\\theta}_2$", alpha=0.7)
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax3.get_yaxis().set_label_coords(-0.11,0.5)
ax3.set_ylabel('Orientation [rad]', fontsize="10" )
ax3.get_xaxis().set_visible(False)

ax4.plot(data['t'], data['fn'], label="$f_{n1}$", alpha=0.7)
ax4.plot(data['t'], data['fn2'], '--', label="$f_{n2}$", alpha=0.7)
ax4.set_xlabel('Time [s]', fontsize="10")
ax4.set_ylabel('Normal force [N]', fontsize="10")
ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax4.get_yaxis().set_label_coords(-0.11,0.5)

plt.tight_layout(h_pad=0.015)
plt.show()
