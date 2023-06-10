import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import frictionModelsCPP.build.FrictionModelCPPClass as cpp
import frictionModelsCPP.build.ReducedFrictionModelCPPClass as red_cpp

import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'

dt = 1e-4
r = 0.05
sim_time = 5

p = {'grid_shape': (21, 21),  # number of grid elements in x any
     'grid_size': r / 21,  # the physical size of each grid element
     'mu_c': 1.0,
     'mu_s': 1.2,
     'v_s': 1e-3,
     'alpha': 2,
     's0': 1e5,
     's1': 2e1,
     's2': 0.2,
     'dt': dt,
     'z_ba_ratio': 0.9,
     'stability': True,
     'elasto_plastic': False,
     'steady_state': False,
     'n_ls': 20}

p_elasto = {'grid_shape': (21, 21),  # number of grid elements in x any
             'grid_size': r / 21,  # the physical size of each grid element
             'mu_c': 1.0,
             'mu_s': 1.2,
             'v_s': 1e-3,
             'alpha': 2,
             's0': 1e5,
             's1': 2e1,
             's2': 0.2,
             'dt': dt,
             'z_ba_ratio': 0.9,
             'stability': True,
             'elasto_plastic': True,
             'steady_state': False,
             'n_ls': 20}

class CircularDisc(object):
    def __init__(self, m, r, dt):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.mass = m
        self.r = r
        self.dt = dt
        self.I = 0.5 * self.mass * self.r**2
        g = 9.81
        self.fn = m*g


    def step(self, fx, fy, ftau):
        ax = fx/self.mass
        ay = fy/self.mass
        atau = ftau/self.I
        acc = np.array([ax, ay, atau])
        self.pos += self.vel * self.dt + 1/2 * acc * self.dt**2
        self.vel += acc * self.dt
        return self.vel, self.pos


def properties_to_list(prop):
    list_ = []
    for index, key in enumerate(prop):
        if key == "grid_shape":
            list_.append(prop[key][0])
            list_.append(prop[key][1])
        else:
            list_.append(prop[key])
    return list_


disc = CircularDisc(1, r, dt)
disc_elasto = CircularDisc(1, r, dt)

fic = cpp.FullFrictionModel()
#fic = red_cpp.ReducedFrictionModel()

fic.init(properties_to_list(p), "Circle", disc.fn)


fic_elasto = cpp.FullFrictionModel()
#fic_elasto = red_cpp.ReducedFrictionModel()

fic_elasto.init(properties_to_list(p_elasto), "Circle", disc_elasto.fn)

num_time_steps = int(sim_time/dt)
f = [0,0,0]
f_e = [0,0,0]

data = np.zeros((10, num_time_steps))

for i_t in tqdm(range(num_time_steps)):
    t = i_t*dt
    if t < 1:
        fx = 1
        if t < 0.2:
            fx = 1 * t/0.2
        fy = 0
        ftau = 0
    elif t< 3:
        fx = 1 + 0.4*np.sin(3*2*np.pi*t)
        fy = 0
        ftau = 0
        if t > 2.8:
            ftau = 0.03 * (t-2.8)/0.2
    else:
        fx = 1 + 0.4 * np.sin(3*2*np.pi*t)
        fy = 0
        ftau = 0.03 + 0.01*np.sin(3*2*np.pi*t)

    vel, pos = disc.step(fx + f[0], fy + f[1], ftau + f[2])
    vel_elasto, pos_elasto = disc_elasto.step(fx + f_e[0], fy + f_e[1], ftau + f_e[2])
    f = fic.step(vel.tolist())
    f_e = fic_elasto.step(vel_elasto.tolist())

    data[0, i_t] = t
    data[1, i_t] = pos[0]
    data[2, i_t] = pos[1]
    data[3, i_t] = pos[2]

    data[4, i_t] = fx
    data[5, i_t] = fy
    data[6, i_t] = ftau

    data[7, i_t] = pos_elasto[0]
    data[8, i_t] = pos_elasto[1]
    data[9, i_t] = pos_elasto[2]

#sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 2})
sns.set_theme("paper", "ticks", font_scale=1.3, rc={"lines.linewidth": 2})
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 12))
sns.despine(f)

ax1.plot(data[0,:], data[1,:], '--', label="$p_x$", alpha=0.7)
ax1.plot(data[0,:], data[2,:], '--', label="$p_y$", alpha=0.7)
ax1.plot(data[0,:], data[7,:], label="$p_x$ elasto-plastic", alpha=0.7)
ax1.plot(data[0,:], data[8,:], label="$p_y$ elasto-plastic", alpha=0.7)
ax1.legend(loc=2)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Position [m]')
ax1.set_title('Position')

ax2.plot(data[0,:], data[3,:], '--', label="$\\Omega$", alpha=0.7)
ax2.plot(data[0,:], data[9,:], label="$\\Omega$ elasto-plastic", alpha=0.7)

ax2.legend(loc=2)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('Orientation [rad]')
ax2.set_title('Orientation')

ax3.plot(data[0,:], data[4,:], label="$f_x$", alpha=0.7)
ax3.plot(data[0,:], data[5,:], label="$f_y$", alpha=0.7)
ax3.legend(loc=2)
ax3.set_xlabel('t [s]')
ax3.set_ylabel('Force [N]')
ax3.set_title('Tangential force')

ax4.plot(data[0,:], data[6,:], label="$f_{\\tau}$", alpha=0.7)
ax4.legend(loc=2)
ax4.set_xlabel('t [s]')
ax4.set_ylabel('Torque [N/m]')
ax4.set_title('Torque')

plt.tight_layout()
plt.show()


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
sns.despine(f)

ax1.plot(data[0,:], data[1,:], '--', label="$p_x$", alpha=0.7)
ax1.plot(data[0,:], data[2,:], '--', label="$p_y$", alpha=0.7)
ax1.plot(data[0,:], data[7,:], label="$p_x$ elasto-plastic", alpha=0.7)
ax1.plot(data[0,:], data[8,:], label="$p_y$ elasto-plastic", alpha=0.7)
ax1.legend(loc=2)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Position [m]')
ax1.set_title('Position')

ax2.plot(data[0,:], data[3,:], '--', label="$\\Omega$", alpha=0.7)
ax2.plot(data[0,:], data[9,:], label="$\\Omega$ elasto-plastic", alpha=0.7)

ax2.legend(loc=2)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('Orientation [rad]')
ax2.set_title('Orientation')


plt.tight_layout()
plt.show()


