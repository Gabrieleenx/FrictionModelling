"""
This file simulates in-hand stick and slip motion by varying the grip force.
"""
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from friction_models_cpp import ModelRed, ModelDist, Force
from matplotlib.patches import Rectangle, Circle
import matplotlib as mpl
from scipy.integrate import ode, solve_ivp
from frictionModels.utils import vel_to_point, move_force_to_point
import surfaces.surfaces as surf
import time
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
shape1 = 'proportional_surface_circle'
shape2 = 'proportional_surface_circle2'
shape3 = 'Circle'

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

        self.y_init = [0.0, 0.0, 0.0, 0.002, 0.0, 0.0]

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

force_red = Force()
force_dist = Force()
force_red2 = Force()
force_dist2 = Force()
force_red3 = Force()
force_dist3 = Force()


time_r_const_p = []
time_d_const_p = []

time_r_scale_p = []
time_d_scale_p = []

time_r_change_p = []
time_d_change_p = []


for k in range(10):
    sim_obj_red = SimObj(ModelRed(shape1, p))
    p_surf = surf.proportional_surface_circle(p['grid_shape'], 10)
    sim_obj_red.friction_model.update_surface(p_surf, 'changing_surface', 1)

    sim_obj_dist = SimObj(ModelDist(shape1, p))

    y_red = sim_obj_red.y_init
    y_red = np.array(y_red)

    y_dist = sim_obj_dist.y_init
    y_dist = np.array(y_dist)


    sim_obj_red2 = SimObj(ModelRed(shape2, p))
    p_surf = surf.proportional_surface_circle2(p['grid_shape'], 10)
    sim_obj_red2.friction_model.update_surface(p_surf, 'changing_surface', 1)

    sim_obj_dist2 = SimObj(ModelDist(shape1, p))

    y_red2 = sim_obj_red2.y_init
    y_red2 = np.array(y_red2)
    y_dist2 = sim_obj_dist2.y_init
    y_dist2 = np.array(y_dist2)

    sim_obj_red3 = SimObj(ModelRed(shape3, p))
    sim_obj_dist3 = SimObj(ModelDist(shape3, p))

    y_red3 = sim_obj_red3.y_init
    y_red3 = np.array(y_red3)
    y_dist3 = sim_obj_dist3.y_init
    y_dist3 = np.array(y_dist3)

    num_time_steps = int(sim_time/dt)





    start_time = time.time()
    for i_t in tqdm(range(num_time_steps)):
        fn = 1.73+0.91*np.sin(i_t*dt*2*np.pi*2) - 0.0*i_t*dt
        p_surf = surf.proportional_surface_circle(p['grid_shape'], fn)
        sim_obj_dist.friction_model.update_surface(p_surf, 'changing_surface')


        sol = solve_ivp(sim_obj_dist.step_ode, (0, dt), y_dist, method='LSODA', t_eval=[dt], args=[fn, force_dist], atol=atol, rtol=rtol, max_step=1e-3)
        y_dist = sol.y[:, -1]  # Update initial conditions for the next step

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_d_scale_p.append(elapsed_time)

    print('Solve time dist', elapsed_time)

    start_time = time.time()
    for i_t in tqdm(range(num_time_steps)):
        fn = 1.73+0.91*np.sin(i_t*dt*2*np.pi*2) - 0.0*i_t*dt
        p_surf = surf.proportional_surface_circle(p['grid_shape'], fn)
        sim_obj_red.friction_model.update_surface(p_surf, 'changing_surface', 0)

        sol = solve_ivp(sim_obj_red.step_ode, (0, dt), y_red, method='LSODA', t_eval=[dt], args=[fn, force_red], atol=atol, rtol=rtol, max_step=1e-3)
        y_red = sol.y[:, -1]  # Update initial conditions for the next step
        y_obj = y_red[len(sim_obj_red.y_fric):]

    end_time = time.time()
    elapsed_time = end_time - start_time

    time_r_scale_p.append(elapsed_time)

    print('Solve time red', elapsed_time)

    start_time = time.time()
    for i_t in tqdm(range(num_time_steps)):
        fn = 1.73+0.91*np.sin(i_t*dt*2*np.pi*2) - 0.0*i_t*dt
        p_surf = surf.proportional_surface_circle2(p['grid_shape'], fn)
        sim_obj_red2.friction_model.update_surface(p_surf, 'changing_surface', 1)
        sol = solve_ivp(sim_obj_red2.step_ode, (0, dt), y_red2, method='LSODA', t_eval=[dt], args=[fn, force_red2], atol=atol, rtol=rtol, max_step=1e-3)
        y_red2 = sol.y[:, -1]  # Update initial conditions for the next step

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_r_change_p.append(elapsed_time)
    print('Solve time red2', elapsed_time)

    start_time = time.time()
    for i_t in tqdm(range(num_time_steps)):
        fn = 1.73+0.91*np.sin(i_t*dt*2*np.pi*2) - 0.0*i_t*dt

        p_surf = surf.proportional_surface_circle2(p['grid_shape'], fn)
        sim_obj_dist2.friction_model.update_surface(p_surf, 'changing_surface')

        sol = solve_ivp(sim_obj_dist2.step_ode, (0, dt), y_dist2, method='LSODA', t_eval=[dt], args=[fn, force_dist2], atol=atol, rtol=rtol, max_step=1e-3)
        y_dist2 = sol.y[:, -1]  # Update initial conditions for the next step

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_d_change_p.append(elapsed_time)
    print('Solve time dist2', elapsed_time)

    start_time = time.time()
    for i_t in tqdm(range(num_time_steps)):
        fn = 1.73+0.91*np.sin(i_t*dt*2*np.pi*2) - 0.0*i_t*dt
        sol = solve_ivp(sim_obj_red3.step_ode, (0, dt), y_red3, method='LSODA', t_eval=[dt], args=[fn, force_red3], atol=atol, rtol=rtol, max_step=1e-3)
        y_red3 = sol.y[:, -1]  # Update initial conditions for the next step

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_r_const_p.append(elapsed_time)
    print('Solve time red3', elapsed_time)

    start_time = time.time()
    for i_t in tqdm(range(num_time_steps)):
        fn = 1.73+0.91*np.sin(i_t*dt*2*np.pi*2) - 0.0*i_t*dt

        sol = solve_ivp(sim_obj_dist3.step_ode, (0, dt), y_dist3, method='LSODA', t_eval=[dt], args=[fn, force_dist3], atol=atol, rtol=rtol, max_step=1e-3)
        y_dist3 = sol.y[:, -1]  # Update initial conditions for the next step

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_d_const_p.append(elapsed_time)
    print('Solve time dist3', elapsed_time)


print('time_r_const_p', np.mean(time_r_const_p))
print('time_d_const_p', np.mean(time_d_const_p))
print('time_r_scale_p', np.mean(time_r_scale_p))
print('time_d_scale_p', np.mean(time_d_scale_p))
print('time_r_change_p', np.mean(time_r_change_p))
print('time_d_change_p', np.mean(time_d_change_p))
