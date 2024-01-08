"""
This test tests the drifting of the planar LuGre model under oscillating loads.
"""
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from friction_models_cpp import ModelRed, ModelDist, Force
import matplotlib as mpl
from scipy.integrate import solve_ivp

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

model = 'Reduced'  # which friction model to test, either 'Distributed' or 'Reduced'
test_type = 'Normal' # i which direction the oscillation is applied, 'Tangential' or 'Normal'

dt = 1e-2  # time step
r = 0.05  # size of contact patch
sim_time = 5  # total sim time
mass = 1
atol = 1e-8
rtol = 1e-6
cells = 21
# Parameters
p = {'grid_shape': (cells, cells),  # number of grid elements in x any
     'grid_size': r / cells,  # the physical size of each grid element
     'mu_c': 1.0,
     'mu_s': 1.2,
     'v_s': 1e-3,
     'alpha': 2,  # called gamma in paper
     's0': 1e6,
     's1': 8e2,
     's2': 0.2,
     'dt': dt,
     'z_ba_ratio': 0.9,
     'stability': False,
     'elasto_plastic': False,
     'steady_state': False,
     'n_ls': 20}

p_elasto = {'grid_shape': (cells, cells),  # number of grid elements in x any
             'grid_size': r / cells,  # the physical size of each grid element
             'mu_c': 1.0,
             'mu_s': 1.2,
             'v_s': 1e-3,
             'alpha': 2,
             's0': 1e6,
             's1': 8e2,
             's2': 0.2,
             'dt': dt,
             'z_ba_ratio': 0.9,
             'stability': False,
             'elasto_plastic': True,
             'steady_state': False,
             'n_ls': 20}

# Class for simulating disc
class CircularDisc(object):
    def __init__(self, m, r):
        self.mass = m
        self.r = r
        self.I = 0.5 * self.mass * self.r**2
        g = 9.81
        self.fn = m*g
        self.y_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def step(self, t, y, f):
        fx, fy, ftau = f[0], f[1], f[2]
        ax = fx/self.mass
        ay = fy/self.mass
        atau = ftau/self.I
        dy = [ax, ay, atau, y[0], y[1], y[2]]
        return dy

class SimObj(object):
    def __init__(self, friction_model):
        self.friction_model = friction_model
        self.obj_dynamics = CircularDisc(mass, r)
        self.y_fric = self.friction_model.y_init
        self.y_object = self.obj_dynamics.y_init
        self.y_init = self.y_fric + self.y_object

    def step_ode(self, t, y, f_n, f_t, force):
        y_f = y[:len(self.y_fric)]
        y_obj = y[len(self.y_fric):]
        v_sensor = [y_obj[0],y_obj[1],y_obj[2]]
        self.friction_model.model_fric.set_fn(f_n)
        dz = self.friction_model.f(t, y_f, v_sensor, force)
        f_ = [force.fx+f_t[0], force.fy+f_t[1], force.tau+f_t[2]]
        dy_obj = self.obj_dynamics.step(t, y_obj, f_)
        return dz + dy_obj



# initialize friction models
if model == 'Distributed':
    fic = SimObj(ModelDist('Circle', p))
    fic_elasto = SimObj(ModelDist('Circle', p_elasto))
else:
    fic = SimObj(ModelRed('Circle', p))
    fic_elasto = SimObj(ModelRed('Circle', p_elasto))

# simulate
num_time_steps = int(sim_time/dt)
data = np.zeros((11, num_time_steps))

y = fic.y_init
y_elasto = fic_elasto.y_init

force = Force()
force_elasto = Force()

for i_t in tqdm(range(num_time_steps)):
    t = i_t*dt
    # force profile
    fn_ = fic.obj_dynamics.fn
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
        fn_ = fic.obj_dynamics.fn
        if t > 1:
            fn_ = fic.obj_dynamics.fn + np.sin(3 * 2 * np.pi * t)

    # step in simulation

    f_t = [fx, fy, ftau]
    sol = solve_ivp(fic.step_ode, (0, dt), y, method='LSODA', t_eval=[dt], args=[fn_, f_t, force], atol=atol,
                    rtol=rtol, max_step=1e-3)
    y = sol.y[:, -1]  # Update initial conditions for the next step

    sol = solve_ivp(fic_elasto.step_ode, (0, dt), y_elasto, method='LSODA', t_eval=[dt], args=[fn_, f_t, force], atol=atol,
                    rtol=rtol, max_step=1e-3)
    y_elasto = sol.y[:, -1]  # Update initial conditions for the next step

    # store data
    data[0, i_t] = t
    y_obj = y[len(fic.y_fric):]
    data[1, i_t] = y_obj[3]
    data[2, i_t] = y_obj[4]
    data[3, i_t] = y_obj[5]

    data[4, i_t] = fx
    data[5, i_t] = fy
    data[6, i_t] = ftau

    y_obj = y_elasto[len(fic_elasto.y_fric):]
    data[7, i_t] = y_obj[3]
    data[8, i_t] = y_obj[4]
    data[9, i_t] = y_obj[5]
    data[10, i_t] = fn_

# plotting
if test_type == 'Tangential':
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 5))
else:
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 4))

ax1.plot(data[0,:], data[1,:], '--', label="$x_d$", alpha=0.7)
ax1.plot(data[0,:], data[2,:], '--', label="$y_d$", alpha=0.7)
ax1.plot(data[0,:], data[7,:], label="$x_d$ e-p", alpha=0.7)
ax1.plot(data[0,:], data[8,:], label="$y_d$ e-p", alpha=0.7)

ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Position [m]', fontsize="10")
ax1.get_xaxis().set_visible(False)

ax2.plot(data[0,:], data[3,:], '--', label="$\\theta_d$", alpha=0.7)
ax2.plot(data[0,:], data[9,:], label="$\\theta_d$ e-p", alpha=0.7)

ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Orientation [rad]', fontsize="10")
ax2.get_xaxis().set_visible(False)
if test_type == 'Tangential':

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
else:
    ax3.plot(data[0,:], data[10,:], label="$f_N$", alpha=0.7)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
    ax3.get_yaxis().set_label_coords(-0.11,0.5)
    ax3.set_ylabel('Force [N]', fontsize="10")
plt.tight_layout(h_pad=0.015)
plt.show()


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3))

ax1.plot(data[0,:], data[1,:], '--', label="$x_r$", alpha=0.7)
ax1.plot(data[0,:], data[2,:], '--', label="$y_r$", alpha=0.7)
ax1.plot(data[0,:], data[7,:], label="$x_r$ e-p", alpha=0.7)
ax1.plot(data[0,:], data[8,:], label="$y_r$ e-p", alpha=0.7)

ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Position [m]', fontsize="10")
ax1.get_xaxis().set_visible(False)

ax2.plot(data[0,:], data[3,:], '--', label="$\\theta_r$", alpha=0.7)
ax2.plot(data[0,:], data[9,:], label="$\\theta_r$ e-p", alpha=0.7)

ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Orientation [rad]', fontsize="10")
ax2.set_xlabel('Time [s]')

plt.tight_layout(h_pad=0.015)
plt.show()


