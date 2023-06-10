import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import frictionModelsCPP.build.FrictionModelCPPClass as cpp
import frictionModelsCPP.build.ReducedFrictionModelCPPClass as red_cpp
from animate import Animate
from frictionModels.utils import vel_to_cop
import surfaces.surfaces as surf
from frictionModels.frictionModel import FullFrictionModel, ReducedFrictionModel
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
p = {'grid_shape': (21, 21),  # number of grid elements in x any
     'grid_size': r / 21,  # the physical size of each grid element
     'mu_c': 1.0,
     'mu_s': 1.2,
     'v_s': 1e-3,
     'alpha': 2,
     's0': 1e6,
     's1': 2e2,
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


class obeject_dynamics(object):
    def __init__(self, h, b, m, dt):
        self.dt = dt
        self.m = m
        self.h = h
        self.b = b
        self.I = (m/12) * (b**2 + h**2)
        self.vel = np.zeros(3)
        self.pos = np.array([0.002,0.0,0.0])

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

object_ = obeject_dynamics(0.15, 0.08, 0.2, dt)
object_red = obeject_dynamics(0.15, 0.08, 0.2, dt)

fic = cpp.FullFrictionModel()
fic.init(properties_to_list(p), "Circle", 2)

fic_red = red_cpp.ReducedFrictionModel()
fic_red.init(properties_to_list(p), "Circle", 2)

num_time_steps = int(sim_time/dt)
f_f = [0,0,0]
f_f_red = [0,0,0]

data = {'p_x':[], 'p_y':[], 'theta':[], 'p_x_red':[], 'p_y_red':[], 'theta_red':[], 't':[], 'fn':[], 'f_x':[], 'f_y':[], 'f_tau':[]}

i_c = int(0.001/dt)
for i_t in tqdm(range(num_time_steps)):
    fn = 1.73+0.9*np.sin(i_t*dt*2*np.pi*2) - 0.0*i_t*dt

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


    if i_t%i_c == 0:
        data['p_x'].append(pos[0])
        data['p_y'].append(pos[1])
        data['theta'].append(pos[2])

        data['p_x_red'].append(pos_red[0])
        data['p_y_red'].append(pos_red[1])
        data['theta_red'].append(pos_red[2])
        data['f_x'].append(f[0])
        data['f_y'].append(f[1])
        data['f_tau'].append(f[2])
        data['t'].append(i_t*dt)
        data['fn'].append(fn)

def plot_box(x, y, theta, h, w, ax, color, label):
    rotation = 180*theta/np.pi
    rect = Rectangle((x - w / 2, y - h / 2), w, h, angle=rotation, facecolor=color, edgecolor=color, fill=False, alpha=0.7, rotation_point='center', label=label)
    ax.add_patch(rect)
    rect.set_linewidth(2)


f, ax = plt.subplots(1, 1, figsize=(6,4))
lim_scale = 0.6
ax.set_xlim(-lim_scale*object_width, lim_scale*object_width)
ax.set_ylim(-object_height, 0.8*object_height)
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
circle = Circle((0,0), 0.02, edgecolor="black", label='Gripper')
circle.set_facecolor((160/256, 138/256, 119/256, 0.5))
ax.add_patch(circle)
circle.set_linewidth(2)
circle2 = Circle((0,0), 0.001, edgecolor="black")
ax.add_patch(circle2)

ax.plot(data['p_x'], data['p_y'], alpha=0.7, label='CoM')
ax.set_xlabel('x [m]', fontsize="10")
ax.set_ylabel('Position y [m]', fontsize="10")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")

plt.tight_layout(h_pad=0.015)
plt.show()

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,4))
#sns.despine(f)

ax1.plot(data['t'], data['p_x'], label="$x$", alpha=0.7)
ax1.plot(data['t'], data['p_y'], label="$y$", alpha=0.7)
ax1.plot(data['t'], data['p_x_red'], '--', label="$\\bar{x}$", alpha=0.7)
ax1.plot(data['t'], data['p_y_red'], '--', label="$\\bar{y}$", alpha=0.7)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Position [m]', fontsize="10" )
ax1.get_xaxis().set_visible(False)

ax2.plot(data['t'], data['theta'], label="$\\theta$", alpha=0.7)
ax2.plot(data['t'], data['theta_red'], '--', label="$\\bar{\\theta}$", alpha=0.7)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Orientation [rad]', fontsize="10" )
ax2.get_xaxis().set_visible(False)

ax3.plot(data['t'], data['fn'], label="$f_n$", alpha=0.7)
ax3.set_xlabel('t [s]', fontsize="10")
ax3.set_ylabel('Normal force [N]', fontsize="10")
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax3.get_yaxis().set_label_coords(-0.11,0.5)

plt.tight_layout(h_pad=0.015)
plt.show()
