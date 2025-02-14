"""
This file simulates in-hand stick and slip motion by varying the grip force.
"""
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from test_ode_solver.friction_models_cpp import ModelRed, ModelDist, Force
from matplotlib.patches import Rectangle, Circle
import matplotlib as mpl
from scipy.integrate import ode, solve_ivp
from frictionModels.utils import vel_to_point, move_force_to_point

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

dt = 1e-2
r = 0.01
sim_time = 2
n = 5
time_stamps = np.arange(n)/(n-1) * (sim_time)
shape = 'Circle'
object_width = 0.15
object_height = 0.08
atol = 1e-8
rtol = 1e-6
p = {'grid_shape': (21, 21),  # number of grid elements in x any
     'grid_size': 2*r / 21,  # the physical size of each grid element
     'mu_c': 1.0,
     'mu_s': 1.2,
     'v_s': 1e-3,
     'alpha': 2, # called gamma in paper
     's0': 1e6,
     's1': 8e2,
     's2': 0.2,
     'dt': dt,
     'z_ba_ratio': 0.9,
     'stability': False,
     'elasto_plastic': True,
     'steady_state': False,
     'n_ls': 20}

class ObjectDynamics(object):
    def __init__(self, h, b, m):
        self.m = m
        self.h = h
        self.b = b
        self.I = (m/12) * (b**2 + h**2)

        self.y_init = [0.0, 0.0, 0.0, 0.025, 0.0, 0.0]

    def step(self, t, y, f_sensor):
        # move to object frame
        f_center = move_force_to_point(f_sensor, y[3:5])
        f_g = -9.81 * self.m
        fx = f_center[0]
        fy = f_center[1] + f_g
        ftau = f_center[2]

        ax = fx/self.m
        ay = fy/self.m
        atau = ftau/self.I
        dy = [ax, ay, atau, y[0], y[1], y[2]]

        return dy

class SimObj(object):
    def __init__(self, friction_model):
        self.friction_model = friction_model
        self.obj_dynamics = ObjectDynamics(0.15, 0.08, 0.2)
        self.y_fric = self.friction_model.y_init
        self.y_object = self.obj_dynamics.y_init
        self.y_init = self.y_fric + self.y_object

    def step_ode(self, t, y, f_n, force):
        y_f = y[:len(self.y_fric)]
        y_obj = y[len(self.y_fric):]
        pos_xy = y_obj[3:5]
        vel_vec = y_obj[:3]
        v_sensor = vel_to_point(-pos_xy, vel_vec)

        self.friction_model.model_fric.set_fn(f_n)
        dz = self.friction_model.f(t, y_f, v_sensor, force)
        force.fx = 2*force.fx
        force.fy = 2 * force.fy
        force.tau = 2 * force.tau
        f_sensor = [force.fx, force.fy, force.tau]
        dy_obj = self.obj_dynamics.step(t, y_obj, f_sensor)
        return dz + dy_obj

    def get_forces(self, t, y, f_n, force):
        # needed to get accurate forces outside of the solver
        y_f = y[:len(self.y_fric)]
        y_obj = y[len(self.y_fric):]
        pos_xy = y_obj[3:5]
        vel_vec = y_obj[:3]
        v_sensor = vel_to_point(-pos_xy, vel_vec)
        self.friction_model.model_fric.set_fn(f_n)
        _ = self.friction_model.f(t, y_f, v_sensor, force)

sim_obj_red = SimObj(ModelRed(shape, p))
sim_obj_dist = SimObj(ModelDist(shape, p))
y_red = sim_obj_red.y_init

y_red = np.array(y_red)
y_red2 = y_red.copy()

y_dist = sim_obj_dist.y_init
y_dist = np.array(y_dist)
y_dist2 = y_dist.copy()

force_red = Force()
force_dist = Force()
force_red2 = Force()
force_dist2 = Force()

num_time_steps = int(sim_time/dt)

data = {'p_x':[], 'p_y':[], 'theta':[],
        'p_x2':[], 'p_y2':[], 'theta2':[],
        'p_x_red':[], 'p_y_red':[], 'theta_red':[],
        'p_x_red2':[], 'p_y_red2':[], 'theta_red2':[],
        't':[], 'fn':[],'fn2':[],
        'f_x_red':[], 'f_y_red':[], 'f_tau_red':[],
        'f_x_red2':[], 'f_y_red2':[], 'f_tau_red2':[],
        'f_x_dist':[], 'f_y_dist':[], 'f_tau_dist':[],
        'f_x_dist2':[], 'f_y_dist2':[], 'f_tau_dist2':[]}

for i_t in tqdm(range(num_time_steps)):
    t = i_t*dt
    fn = 3.6+0.9*np.sin(i_t*dt*2*np.pi*2) - 0.0*i_t*dt
    fn2 = 4 # do a second force profile that gives more rotation
    if 0.6*(num_time_steps/4) < i_t and i_t < (num_time_steps/4):
        fn2=2.9

    sol = solve_ivp(sim_obj_red.step_ode, (0, dt), y_red, method='LSODA', t_eval=[dt], args=[fn, force_red], atol=atol, rtol=rtol, max_step=1e-3)
    y_red = sol.y[:, -1]  # Update initial conditions for the next step
    y_obj = y_red[len(sim_obj_red.y_fric):]
    _ = sim_obj_red.get_forces(t,y_red, fn, force_red)

    sol = solve_ivp(sim_obj_red.step_ode, (0, dt), y_red2, method='LSODA', t_eval=[dt], args=[fn2, force_red2], atol=atol, rtol=rtol, max_step=1e-3)
    y_red2 = sol.y[:, -1]  # Update initial conditions for the next step
    y_obj2 = y_red2[len(sim_obj_red.y_fric):]
    _ = sim_obj_red.get_forces(t,y_red2, fn2, force_red2)

    data['p_x_red'].append(y_obj[3])
    data['p_y_red'].append(y_obj[4])
    data['theta_red'].append(y_obj[5])

    data['p_x_red2'].append(y_obj2[3])
    data['p_y_red2'].append(y_obj2[4])
    data['theta_red2'].append(y_obj2[5])

    data['f_x_red'].append(force_red.fx)
    data['f_y_red'].append(force_red.fy)
    data['f_tau_red'].append(force_red.tau)

    data['f_x_red2'].append(force_red2.fx)
    data['f_y_red2'].append(force_red2.fy)
    data['f_tau_red2'].append(force_red2.tau)

    data['t'].append(i_t*dt)
    data['fn'].append(fn)
    data['fn2'].append(fn2)


for i_t in tqdm(range(num_time_steps)):
    t = i_t*dt
    fn = 3.6+0.9*np.sin(i_t*dt*2*np.pi*2) - 0.0*i_t*dt
    fn2 = 4 # do a second force profile that gives more rotation
    if 0.6*(num_time_steps/4) < i_t and i_t < (num_time_steps/4):
        fn2=2.9

    #sol = solve_ivp(sim_obj.step_ode, (0, dt), y, t_eval=[dt], args=[fn, force], atol=1e-11, rtol=1e-8)
    sol = solve_ivp(sim_obj_dist.step_ode, (0, dt), y_dist, method='LSODA', t_eval=[dt], args=[fn, force_dist], atol=atol, rtol=rtol, max_step=1e-3)
    y_dist = sol.y[:, -1]  # Update initial conditions for the next step
    y_obj = y_dist[len(sim_obj_dist.y_fric):]
    _ = sim_obj_dist.get_forces(t,y_dist, fn, force_dist)

    sol = solve_ivp(sim_obj_dist.step_ode, (0, dt), y_dist2, method='LSODA', t_eval=[dt], args=[fn2, force_dist2], atol=atol, rtol=rtol, max_step=1e-3)
    y_dist2 = sol.y[:, -1]  # Update initial conditions for the next step
    y_obj2 = y_dist2[len(sim_obj_dist.y_fric):]
    _ = sim_obj_dist.get_forces(t,y_dist2, fn2, force_dist2)

    data['p_x'].append(y_obj[3])
    data['p_y'].append(y_obj[4])
    data['theta'].append(y_obj[5])

    data['f_x_dist'].append(force_dist.fx)
    data['f_y_dist'].append(force_dist.fy)
    data['f_tau_dist'].append(force_dist.tau)

    data['p_x2'].append(y_obj2[3])
    data['p_y2'].append(y_obj2[4])
    data['theta2'].append(y_obj2[5])

    data['f_x_dist2'].append(force_dist2.fx)
    data['f_y_dist2'].append(force_dist2.fy)
    data['f_tau_dist2'].append(force_dist2.tau)



def plot_box(x, y, theta, h, w, ax, color, label):
    rotation = 180*theta/np.pi
    rect = Rectangle((x - w / 2, y - h / 2), w, h, angle=rotation, facecolor=color, edgecolor=color, fill=False, alpha=0.7, rotation_point='center', label=label)
    ax.add_patch(rect)
    rect.set_linewidth(2)


f, ax = plt.subplots(1, 1, figsize=(4.2,2.6))
lim_scale = 0.6
ax.set_xlim(-lim_scale*object_width, lim_scale*object_width)
ax.set_ylim(-object_height, 0.8*object_height)
i_t_max = int(sim_time/(dt))
for t in time_stamps:
    if t>=sim_time:
        t=sim_time-dt
    i_t = int(t/(dt))

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
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")

plt.tight_layout(h_pad=0.015)
plt.show()

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6,5))
#f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,3.8))

ax1.plot(data['t'], data['p_x'], label="$x_{d,1}$", alpha=0.7)
ax1.plot(data['t'], data['p_x_red'], label="$x_{r,1}$", alpha=0.7)
ax1.plot(data['t'], data['p_x2'], '--', label="$x_{d,2}$", alpha=0.7)
ax1.plot(data['t'], data['p_x_red2'], '--', label="$x_{r,2}$", alpha=0.7)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Position [m]', fontsize="10" )
ax1.get_xaxis().set_visible(False)

ax2.plot(data['t'], data['p_y'], label="$y_{d,1}$", alpha=0.7)
ax2.plot(data['t'], data['p_y_red'], label="$y_{r,1}$", alpha=0.7)
ax2.plot(data['t'], data['p_y2'], '--', label="$y_{d,2}$", alpha=0.7)
ax2.plot(data['t'], data['p_y_red2'], '--', label="$y_{r,2}$", alpha=0.7)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Position [m]', fontsize="10" )
ax2.get_xaxis().set_visible(False)

ax3.plot(data['t'], data['theta'], label="$\\theta_{d,1}$", alpha=0.7)
ax3.plot(data['t'], data['theta_red'], label="$\\theta}_{r,1}$", alpha=0.7)
ax3.plot(data['t'], data['theta2'], '--', label="$\\theta_{d,2}$", alpha=0.7)
ax3.plot(data['t'], data['theta_red2'], '--', label="$\\theta}_{r,2}$", alpha=0.7)
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax3.get_yaxis().set_label_coords(-0.11,0.5)
ax3.set_ylabel('Orientation [rad]', fontsize="10" )
ax3.set_xlabel('Time [s]', fontsize="10")

ax3.get_xaxis().set_visible(False)
ax4.plot(data['t'], data['fn'], label="$f_{N,1}$", alpha=0.7)
ax4.plot(data['t'], data['fn2'], '--', label="$f_{N,2}$", alpha=0.7)
ax4.set_xlabel('Time [s]', fontsize="10")
ax4.set_ylabel('Normal force [N]', fontsize="10")
ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax4.get_yaxis().set_label_coords(-0.11, 0.5)


plt.tight_layout(h_pad=0.015)
plt.show()

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,4))

ax1.plot(data['t'], data['f_x_red'], label="$f_{r,x,1}$", alpha=0.7)
ax1.plot(data['t'], data['f_x_dist'], label="$f_{d,x,1}$", alpha=0.7)
ax1.plot(data['t'], data['f_x_dist2'], '--', label="$f_{d,x,2}$", alpha=0.7)
ax1.plot(data['t'], data['f_x_red2'], '--', label="$f_{r,x,2}$", alpha=0.7)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax1.set_ylabel('Force [N]', fontsize="10" )
ax1.get_xaxis().set_visible(False)

ax2.plot(data['t'], data['f_y_red'], label="$f_{r,y,1}$", alpha=0.7)
ax2.plot(data['t'], data['f_y_dist'], label="$f_{d,y,1}$", alpha=0.7)
ax2.plot(data['t'], data['f_y_dist2'], '--', label="$f_{d,y,2}$", alpha=0.7)
ax2.plot(data['t'], data['f_y_red2'], '--', label="$f_{r,y,2}$", alpha=0.7)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Force [N]', fontsize="10" )
ax2.get_xaxis().set_visible(False)

ax3.plot(data['t'], data['f_tau_red'], label="$\\tau_{r,1}$", alpha=0.7)
ax3.plot(data['t'], data['f_tau_dist'], label="$\\tau_{d,1}$", alpha=0.7)
ax3.plot(data['t'], data['f_tau_dist2'], '--', label="$\\tau_{d,2}$", alpha=0.7)
ax3.plot(data['t'], data['f_tau_red2'], '--', label="$\\tau_{r,2}$", alpha=0.7)
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax3.get_yaxis().set_label_coords(-0.11,0.5)
ax3.set_ylabel('Torque [Nm]', fontsize="10" )
#ax3.get_xaxis().set_visible(False)

#ax4.plot(data['t'], data['fn'], label="$f_{N,1}$", alpha=0.7)
#ax4.plot(data['t'], data['fn2'], '--', label="$f_{N,2}$", alpha=0.7)
#ax4.set_xlabel('Time [s]', fontsize="10")
#ax4.set_ylabel('Normal force [N]', fontsize="10")
#ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
#ax4.get_yaxis().set_label_coords(-0.11,0.5)

plt.tight_layout(h_pad=0.015)
plt.show()
