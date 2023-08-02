"""
This test tests the drifting of the planar LuGre model under oscillating loads.
"""
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
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

model = 'Reduced'  # which friction model to test, either 'Distributed' or 'Reduced'
test_type = 'Tangential' # i which direction the oscillation is applied, 'Tangential' or 'Normal'

dt = 1e-4  # time step
r = 0.05  # size of contact patch
sim_time = 5  # total sim time

# Parameters
p = {'grid_shape': (21, 21),  # number of grid elements in x any
     'grid_size': r / 21,  # the physical size of each grid element
     'mu_c': 1.0,
     'mu_s': 1.2,
     'v_s': 1e-3,
     'alpha': 2,  # called gamma in paper
     's0': 1e6,
     's1': 8e1,
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
             's0': 1e6,
             's1': 8e1,
             's2': 0.2,
             'dt': dt,
             'z_ba_ratio': 0.9,
             'stability': True,
             'elasto_plastic': True,
             'steady_state': False,
             'n_ls': 20}

# Class for simulating disc
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

# Dict to list for cpp models
def properties_to_list(prop):
    list_ = []
    for index, key in enumerate(prop):
        if key == "grid_shape":
            list_.append(prop[key][0])
            list_.append(prop[key][1])
        else:
            list_.append(prop[key])
    return list_

# initiate sim environments
disc = CircularDisc(1, r, dt)
disc_elasto = CircularDisc(1, r, dt)

# initialize friction models
if model == 'Distributed':
    fic = cpp.DistributedFrictionModel()
    fic_elasto = cpp.DistributedFrictionModel()
else:
    fic = red_cpp.ReducedFrictionModel()
    fic_elasto = red_cpp.ReducedFrictionModel()

fic.init(properties_to_list(p), "Circle", disc.fn)
fic_elasto.init(properties_to_list(p_elasto), "Circle", disc_elasto.fn)

# simulate
num_time_steps = int(sim_time/dt)
f = [0,0,0]
f_e = [0,0,0]
data = np.zeros((10, num_time_steps))

for i_t in tqdm(range(num_time_steps)):
    t = i_t*dt
    # force profile
    if test_type == 'Tangential':
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
    else:
        if t < 1:
            fx = 1
            ftau = 0.03
            if t < 0.2:
                fx = 1 * t / 0.2
                ftau = 0.03 * t / 0.2
            fy = 0
        else:
            fx = 1
            fy = 0
            ftau = 0.03
        fn_ = disc.fn
        if t > 1:
            fn_ = disc.fn + np.sin(3 * 2 * np.pi * t)
            fic.set_fn(fn_)
            fic_elasto.set_fn(fn_)
    # step in simulation
    vel, pos = disc.step(fx + f[0], fy + f[1], ftau + f[2])
    vel_elasto, pos_elasto = disc_elasto.step(fx + f_e[0], fy + f_e[1], ftau + f_e[2])
    f = fic.step(vel.tolist())
    f_e = fic_elasto.step(vel_elasto.tolist())

    # store data
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

# plotting
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 5))

ax1.plot(data[0,:], data[1,:], '--', label="$x$", alpha=0.7)
ax1.plot(data[0,:], data[2,:], '--', label="$y$", alpha=0.7)
ax1.plot(data[0,:], data[7,:], label="$x$ e-p", alpha=0.7)
ax1.plot(data[0,:], data[8,:], label="$y$ e-p", alpha=0.7)

ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Position [m]', fontsize="10")
ax1.get_xaxis().set_visible(False)

ax2.plot(data[0,:], data[3,:], '--', label="$\\theta$", alpha=0.7)
ax2.plot(data[0,:], data[9,:], label="$\\theta$ e-p", alpha=0.7)

ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Orientation [rad]', fontsize="10")
ax2.get_xaxis().set_visible(False)

ax3.plot(data[0,:], data[4,:], label="$f_x$", alpha=0.7)
ax3.plot(data[0,:], data[5,:], label="$f_y$", alpha=0.7)
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax3.get_yaxis().set_label_coords(-0.11,0.5)
ax3.set_ylabel('Force [N]', fontsize="10")
ax3.get_xaxis().set_visible(False)

ax4.plot(data[0,:], data[6,:], label="$\\tau$", alpha=0.7)
ax4.set_xlabel('Time [s]')
ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax4.get_yaxis().set_label_coords(-0.11,0.5)
ax4.set_ylabel('Torque [Nm]', fontsize="10")

plt.tight_layout(h_pad=0.015)
plt.show()


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3))

ax1.plot(data[0,:], data[1,:], '--', label="$\\bar{x}$", alpha=0.7)
ax1.plot(data[0,:], data[2,:], '--', label="$\\bar{y}$", alpha=0.7)
ax1.plot(data[0,:], data[7,:], label="$\\bar{x}$ e-p", alpha=0.7)
ax1.plot(data[0,:], data[8,:], label="$\\bar{y}$ e-p", alpha=0.7)

ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Position [m]', fontsize="10")
ax1.get_xaxis().set_visible(False)

ax2.plot(data[0,:], data[3,:], '--', label="$\\bar{\\theta}$", alpha=0.7)
ax2.plot(data[0,:], data[9,:], label="$\\bar{\\theta}$ e-p", alpha=0.7)

ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Orientation [rad]', fontsize="10")
ax2.set_xlabel('Time [s]')

plt.tight_layout(h_pad=0.015)
plt.show()


