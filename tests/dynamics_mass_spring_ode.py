import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp

import surfaces.surfaces as surf
from velocity_profiles import *
from frictionModels.frictionModel import FullFrictionModel, ReducedFrictionModel, LuGre1D
#from ode_sim import ODEsim


def flatten(t):
    return [item for sublist in t for item in sublist]


class Sim(object):
    def __init__(self, dt, friction_model, object_model):
        self.friction_model = friction_model
        self.object_model = object_model
        self.len_f = 0
        self.len_o = 0
        self.y = self.get_y0()
        self.t = 0
        self.args = []
        self.dt = dt

    def step(self):
        sol = solve_ivp(fun=self.model, t_span=[self.t, self.t+self.dt], y0=self.y, method='RK45',
                        args=self.args, dense_output=False, t_eval=[self.t+self.dt])

        self.y = flatten(sol.y)
        self.t += self.dt

    def model(self, t, y):
        """
        The combined model for the ODE solver
        :param t: time [s]
        :param y: state space
        :return: dy
        """
        y_f = y[0:self.len_f]
        vel = [y[self.len_f], 0, 0]
        dy_f, f = self.friction_model.ode_step(t, y_f, vel)

        y_o = y[self.len_f:self.len_f+self.len_o]

        dy_o = self.object_model.ode_step(t, y_o, f)

        dy = dy_f + dy_o
        return dy

    def get_y0(self):
        """
        Initialize the state space
        :return: y0
        """
        y0_f = self.friction_model.ode_init()
        y0_o = self.object_model.ode_init()
        self.len_f = len(y0_f)
        self.len_o = len(y0_o)
        y0 = y0_f + y0_o
        return y0

    def get_output(self):
        y_o = self.y[self.len_f:self.len_f+self.len_o]
        v_m = y_o[0]
        x_m = y_o[1]
        x_s = y_o[2]
        v = self.object_model.v(self.t)

        y_f = self.y[0:self.len_f]
        vel = [self.y[self.len_f], 0, 0]
        dy_f, f = self.friction_model.ode_step(self.t, y_f, vel)

        out = {'t': self.t, 'x_m': x_m, 'v_m': v_m, 'f_x': f['x'], 'v': v, 'x_s': x_s}
        return out


class Sim_all(object):
    def __init__(self, t_end, friction_model, object_model):
        self.friction_model = friction_model
        self.object_model = object_model
        self.len_f = 0
        self.len_o = 0
        self.y = self.get_y0()
        self.t = 0
        self.t_end = t_end
        self.args = []

    def sim(self):
        sol = solve_ivp(fun=self.model, t_span=[0, self.t_end], y0=self.y, method='RK45',
                        args=self.args, dense_output=False, rtol=1e-5, atol=1e-10)
        print(sol)
        return sol.y, sol.t

    def model(self, t, y):
        """
        The combined model for the ODE solver
        :param t: time [s]
        :param y: state space
        :return: dy
        """
        y_f = y[0:self.len_f]
        vel = [y[self.len_f], 0, 0]
        dy_f, f = self.friction_model.ode_step(t, y_f, vel)

        y_o = y[self.len_f:self.len_f+self.len_o]

        dy_o = self.object_model.ode_step(t, y_o, f)

        dy = dy_f + dy_o
        return dy

    def get_y0(self):
        """
        Initialize the state space
        :return: y0
        """
        y0_f = self.friction_model.ode_init()
        y0_o = self.object_model.ode_init()
        self.len_f = len(y0_f)
        self.len_o = len(y0_o)
        y0 = y0_f + y0_o
        return y0

    def get_output(self):
        y_o = self.y[self.len_f:self.len_f+self.len_o]
        v_m = y_o[0]
        x_m = y_o[1]
        x_s = y_o[2]
        v = self.object_model.v(self.t)

        y_f = self.y[0:self.len_f]
        vel = [self.y[self.len_f], 0, 0]
        dy_f, f = self.friction_model.ode_step(self.t, y_f, vel)

        out = {'t': self.t, 'x_m': x_m, 'v_m': v_m, 'f_x': f['x'], 'v': v, 'x_s': x_s}
        return out


class MassSprringModel(object):
    def __init__(self, m, k):
        self.k_spring = k
        self.m = m

    def ode_init(self):
        """
        Generates initial values for the states.
        :return: y0
        """
        y0 = [0, 0, 0]
        return y0

    def ode_step(self, t, y, f):
        """
        Generates the derivatives for ode solver
        :param t: time [s]
        :param y: list with states
        :param f: friction force, set {'x', 'y', 'tau'} [N, N, Nm]
        :return: dy
        """
        dx_s = self.v(t)
        vx_m = y[0]
        x_m = y[1]
        x_s = y[2]
        f_spring = self.k_spring * (x_s - x_m)

        f_tot = f_spring + f['x']
        a = f_tot / self.m

        dvx_m = a
        dx_m = vx_m

        dy = [dvx_m, dx_m, dx_s]
        return dy

    def v(self, t):
        """
        Velocity of spring
        :param t: time [s]
        :return: v [m/s]
        """
        return 1e-2


properties = {'grid_shape': (20, 20),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.3,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e6,
              's1': 8e1,
              's2': 0.4,
              'dt': 5e-4,
              'stability': True,
              'elasto_plastic': True,
              'z_ba_ratio': 0.9,
              'steady_state': False}

mass = 1
# initialize friction models
shape = surf.PObject(properties['grid_size'], properties['grid_shape'], surf.p_square)
shape.set_fn(mass * 9.81)

planar_lugre = FullFrictionModel(properties=properties)
planar_lugre.update_p_x_y(shape)


obj_model = MassSprringModel(m=mass, k=1e3)

sim = Sim(dt=properties['dt'], friction_model=planar_lugre, object_model=obj_model)

time = 2

"""
sim2 = Sim_all(t_end=time, friction_model=planar_lugre, object_model=obj_model)
# plotting
y,t = sim2.sim()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(t, y[:][sim2.len_f+1], alpha=0.7, label='x mass')
ax1.plot(t, y[:][sim2.len_f+2], alpha=0.7, label='x spring')
ax1.set_title('Distance')
ax1.set_xlabel('Time')
ax1.set_ylabel('Dist [m]')
ax1.legend()

ax2.plot(t, y[sim2.len_f], label='v mass')
ax2.set_title('Velocity')
ax2.set_xlabel('Time')
ax2.set_ylabel('Vel [m/s]')
ax2.legend()

plt.tight_layout()
plt.show()
"""


n_steps = int(time / properties['dt'])

data_full = np.zeros((6, n_steps))  #t, x_mass, v_mass, f_f, v_sprig, x_spring
for i in tqdm(range(n_steps)):
    sim.step()
    out = sim.get_output()
    data_full[0, i] = out['t']
    data_full[1, i] = out['x_m']
    data_full[2, i] = out['v_m']
    data_full[3, i] = out['f_x']
    data_full[4, i] = out['v']
    data_full[5, i] = out['x_s']


obj_model_red = MassSprringModel(m=mass, k=1e3)

planar_lugre_reduced = ReducedFrictionModel(properties=properties)
planar_lugre_reduced.update_p_x_y(shape)
planar_lugre_reduced.update_pre_compute()

sim_red = Sim(dt=properties['dt'], friction_model=planar_lugre_reduced, object_model=obj_model_red)

data_red = np.zeros((6, n_steps))  #t, x_mass, v_mass, f_f, v_sprig, x_spring

for i in tqdm(range(n_steps)):
    sim_red.step()
    out = sim_red.get_output()
    data_red[0, i] = out['t']
    data_red[1, i] = out['x_m']
    data_red[2, i] = out['v_m']
    data_red[3, i] = out['f_x']
    data_red[4, i] = out['v']
    data_red[5, i] = out['x_s']


f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

ax1.plot(data_full[0, :], data_full[1, :], alpha=0.7, label='x mass')
ax1.plot(data_red[0, :], data_red[1, :], '--', label='x mass red')
ax1.plot(data_full[0, :], data_full[5, :], alpha=0.7, label='x spring')
ax1.set_title('Distance')
ax1.set_xlabel('Time')
ax1.set_ylabel('Dist [m]')
ax1.legend()

ax2.plot(data_full[0, :], data_full[2, :], label='v mass')
ax2.plot(data_full[0, :], data_full[4, :], label='v spring')
ax2.plot(data_red[0, :], data_red[2, :], '--', label='v mass red')

ax2.set_title('Velocity')
ax2.set_xlabel('Time')
ax2.set_ylabel('Vel [m/s]')
ax2.legend()

ax3.plot(data_full[0, :], data_full[3, :], label='f friction')
ax3.plot(data_red[0, :], data_red[3, :], label='f friction red')

ax3.set_title('Friction')
ax3.set_xlabel('Time')
ax3.set_ylabel('Friction [N]')
ax3.legend()


plt.tight_layout()
plt.show()

