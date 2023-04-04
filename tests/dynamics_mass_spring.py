import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import surfaces.surfaces as surf
from velocity_profiles import *
from frictionModels.frictionModel import FullFrictionModel, ReducedFrictionModel, LuGre1D

class MassSpringSim(object):
    def __init__(self, dt):
        self.x_mass = 0
        self.v_mass = 0
        self.x_spring = 0
        self.k_spring = 1e3
        self.m = 1
        self.dt = dt

    def step(self, v, f_friction):
        self.x_spring += v*self.dt

        f_spring = self.k_spring*(self.x_spring - self.x_mass)
        f_tot = f_spring + f_friction
        a = f_tot/self.m
        self.v_mass += a*self.dt
        self.x_mass += self.v_mass * self.dt - 1/2 * a*self.dt**2



properties = {'grid_shape': (20, 20),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.3,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e5,
              's1': 2e1,
              's2': 0.4,
              'dt': 1e-4,
              'stability': True,
              'elasto_plastic': True,
              'z_ba_ratio': 0.9,
              'steady_state': False}

mass = 1

time = 3
v = 1e-2

n_steps = int(time / properties['dt'])

# initialize friction models
shape = surf.PObject(properties['grid_size'], properties['grid_shape'], surf.p_square)
shape.set_fn(mass * 9.81)

planar_lugre = FullFrictionModel(properties=properties)
planar_lugre.update_p_x_y(shape)

planar_lugre_reduced = ReducedFrictionModel(properties=properties)
planar_lugre_reduced.update_p_x_y(shape)
planar_lugre_reduced.update_pre_compute()

sim_1 = MassSpringSim(properties['dt'])


data_full = np.zeros((6, n_steps))  #t, x_mass, v_mass, f_f, v_sprig, x_spring

for i in tqdm(range(n_steps)):
    dx = sim_1.v_mass
    vel = {'x': dx, 'y':0, 'tau':0}
    f_friction = planar_lugre.step(vel)
    sim_1.step(v, f_friction['x'])
    data_full[0, i] = i * properties['dt']
    data_full[1, i] = sim_1.x_mass
    data_full[2, i] = sim_1.v_mass
    data_full[3, i] = f_friction['x']
    data_full[4, i] = v
    data_full[5, i] = sim_1.x_spring

sim_2 = MassSpringSim(properties['dt'])

data_red = np.zeros((6, n_steps))  #t, x_mass, v_mass, f_f, v_sprig, x_spring

for i in tqdm(range(n_steps)):
    dx = sim_2.v_mass
    vel = {'x': dx, 'y':0, 'tau':0}
    f_friction = planar_lugre_reduced.step(vel)
    sim_2.step(v, f_friction['x'])
    data_red[0, i] = i * properties['dt']
    data_red[1, i] = sim_2.x_mass
    data_red[2, i] = sim_2.v_mass
    data_red[3, i] = f_friction['x']
    data_red[4, i] = v
    data_red[5, i] = sim_2.x_spring


sim_3 = MassSpringSim(properties['dt'])

data_1d = np.zeros((6, n_steps))  #t, x_mass, v_mass, f_f, v_sprig, x_spring
lugre1d = LuGre1D(properties=properties, fn=mass*9.81)
z = 0
for i in tqdm(range(n_steps)):
    dx = sim_3.v_mass
    dz, f_friction = lugre1d.ode_step(0, [z], [dx])
    z += dz[0] * properties['dt']
    sim_3.step(v, f_friction['x'])
    data_1d[0, i] = i * properties['dt']
    data_1d[1, i] = sim_3.x_mass
    data_1d[2, i] = sim_3.v_mass
    data_1d[3, i] = f_friction['x']
    data_1d[4, i] = v
    data_1d[5, i] = sim_3.x_spring


# plotting

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

ax1.plot(data_full[0, :], data_full[1, :], alpha=0.7, label='x mass')
ax1.plot(data_red[0, :], data_red[1, :], '--', label='x mass red')
ax1.plot(data_full[0, :], data_full[5, :], alpha=0.7, label='x spring')
ax1.plot(data_1d[0, :], data_1d[1, :], '--', label='x LuGre 1d')

ax1.set_title('Distance')
ax1.set_xlabel('Time')
ax1.set_ylabel('Dist [m]')
ax1.legend()

ax2.plot(data_full[0, :], data_full[2, :], label='v mass')
ax2.plot(data_full[0, :], data_full[4, :], label='v spring')
ax2.plot(data_red[0, :], data_red[2, :], '--', label='v mass red')
ax2.plot(data_1d[0, :], data_1d[2, :], '--', label='v LuGre 1d')

ax2.set_title('Velocity')
ax2.set_xlabel('Time')
ax2.set_ylabel('Vel [m/s]')
ax2.legend()

ax3.plot(data_full[0, :], data_full[3, :], label='f friction')
ax3.plot(data_red[0, :], data_red[3, :], label='f friction red')
ax3.plot(data_1d[0, :], data_1d[3, :], label='f LuGre 1d')

ax3.set_title('Friction')
ax3.set_xlabel('Time')
ax3.set_ylabel('Friction [N]')
ax3.legend()


plt.tight_layout()
plt.show()
