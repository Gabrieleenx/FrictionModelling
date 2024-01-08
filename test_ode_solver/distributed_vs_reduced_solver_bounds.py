"""
This file compares the distributed and reduced models.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl
from tests.velocity_profiles import *
from friction_models_cpp import ModelRed, ModelDist, Force
from frictionModels.frictionModel import ReducedFrictionModel
from frictionModels.utils import vel_to_cop
from surfaces.surfaces import non_convex_1, non_convex_2
import surfaces.surfaces as surf
from scipy.integrate import ode, solve_ivp

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0.01
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

fn = 1.0
time = 5
r = 0.01
cells = 21
atol = 1e-8
rtol = 1e-6

properties = {'grid_shape': (cells, cells),  # number of grid elements in x any
              'grid_size': 2*r/cells,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.2,
              'v_s': 1e-3,
              'alpha': 2,  # called gamma in paper
              's0': 1e6,
              's1': 8e2,
              's2': 0.2,
              'dt': 0.01,
              'z_ba_ratio': 0.9,
              'stability': False,
              'elasto_plastic': True,
              'steady_state': False,
              'n_ls': 20}

shape_set = {'Square': surf.p_square, 'Circle': surf.p_circle, 'Line': surf.p_line, 'LineGrad': surf.p_line_grad,
             'NonConvex1':non_convex_1, 'NonConvex2':non_convex_2, 'LineGrad2': surf.p_line_grad, 'LineGrad3': surf.p_line_grad3,
             'LineGrad4': surf.p_line_grad4}

class ModelEllipse():
    def __init__(self, shape_):
        self.ellipse_red = ReducedFrictionModel(properties=properties, ls_active=False)
        shape_ = surf.PObject(properties['grid_size'], properties['grid_shape'], shape_set[shape])
        self.ellipse_red.update_p_x_y(shape_)
        self.ellipse_red.update_pre_compute()
        self.y_init = [0.0, 0.0, 0.0]

    def f(self, t, y, vel, force):
        dy, _ = self.ellipse_red.ode_step(t, y, vel)

        f = self.ellipse_red.force_at_cop
        force.fx_cop = f['x']
        force.fy_cop = f['y']
        force.tau_cop = f['tau']
        return dy

    def update_surface(self, p_surf, name):
        shape_ = surf.PObject(properties['grid_size'], properties['grid_shape'], p_surf)
        self.ellipse_red.update_p_x_y(shape_)
        self.ellipse_red.update_pre_compute()


n_steps = int(time / properties['dt'])

shapes = ['Circle', 'Square', 'Line', 'LineGrad', 'NonConvex1', 'NonConvex2']

error_list_ft = []
error_list_tau = []
error_list_ellipse_ft = []
error_list_ellipse_tau = []
for shape in shapes:
    model_red = ModelRed(shape, properties)
    model = ModelDist(shape, properties)
    model_ellipse = ModelEllipse(shape)
    if shape == 'NonConvex1':
        p_surf = non_convex_1(properties["grid_shape"], 1)
        model.update_surface(p_surf, "non_convex")
        model_red.update_surface(p_surf, "non_convex")

    if shape == 'NonConvex2':
        p_surf = non_convex_2(properties["grid_shape"], 1)
        model.update_surface(p_surf, "non_convex")
        model_red.update_surface(p_surf, "non_convex")

    cop = np.array(model.model_fric.get_cop())
    cop_red = np.array(model_red.model_fric.get_cop())
    print(cop_red)

    y = model.y_init
    y_red = model_red.y_init
    y_ellipse = model_ellipse.y_init

    force = Force()
    force_red = Force()
    force_ellipse = Force()

    dt = properties['dt']
    # running simulation
    error_ft = []
    error_tau = []
    error_el_ft = []
    error_el_tau = []
    for i in tqdm(range(n_steps)):
        t = (i) * properties['dt']
        vel = vel_num_cells(t)
        vel_red = vel_to_cop(-cop_red, vel_vec=vel)
        vel = vel_to_cop(-cop, vel_vec=vel)

        vel_ = [vel['x'], vel['y'], vel['tau']]
        vel_red_ = [vel_red['x'], vel_red['y'], vel_red['tau']]

        sol = solve_ivp(model.f, (0, dt), y, method='LSODA', t_eval=[dt], args=[vel_, force], atol=atol, rtol=rtol, max_step=1e-3)
        y = sol.y[:, -1]  # Update initial conditions for the next step
        _ = model.f(t, y, vel_, force) # to get the correct force
        sol_red = solve_ivp(model_red.f, (0, dt), y_red, method='LSODA', t_eval=[dt], args=[vel_red_, force_red], atol=atol, rtol=rtol, max_step=1e-3)
        y_red = sol_red.y[:, -1]  # Update initial conditions for the next step
        _ = model_red.f(t, y_red, vel_red_, force_red)  # to get the correct force

        sol_el = solve_ivp(model_ellipse.f, (0, dt), y_ellipse, method='LSODA', t_eval=[dt], args=[vel_red_, force_ellipse],
                            atol=atol, rtol=rtol, max_step=1e-3)
        y_ellipse = sol_el.y[:, -1]  # Update initial conditions for the next step
        _ = model_ellipse.f(t, y_ellipse, vel_red_, force_ellipse)

        error_ft.append(np.linalg.norm([force.fx_cop - force_red.fx_cop, force.fy_cop - force_red.fy_cop]))
        error_tau.append(np.linalg.norm([force.tau_cop - force_red.tau_cop]))

        error_el_ft.append(np.linalg.norm([force.fx_cop - force_ellipse.fx_cop, force.fy_cop - force_ellipse.fy_cop]))
        error_el_tau.append(np.linalg.norm([force.tau_cop - force_ellipse.tau_cop]))
    print(np.max(np.abs(error_el_tau)))
    error_list_ft.append(np.array(error_ft))
    error_list_tau.append(np.array(error_tau))
    error_list_ft.append(np.array(error_el_ft))
    error_list_tau.append(np.array(error_el_tau))
# plotting

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
# Generate some example data

positions = []
pos = 0
labels = []
for shape in shapes:
    positions.append(pos)
    positions.append(pos + 2)
    labels.append(shape)
    labels.append(shape + " ellipse")
    pos += 5
# Create a boxplot
color = "firebrick"
VP = ax1.boxplot(error_list_ft, positions=positions, widths=1.5, patch_artist=True,
                showmeans=False, showfliers=False,
                 medianprops={"color": color, "linewidth": 0.5, "alpha": 0.7},
                 boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.5, "alpha": 0.3},
                 whiskerprops={"color": color, "linewidth": 1.5, "alpha": 0.7},
                 capprops={"color": color, "linewidth": 1.5, "alpha": 0.7}, labels=labels)
ax1.set_ylabel('Error $||f_t||$', fontsize="10" )
ax1.get_xaxis().set_visible(False)
ax1.yaxis.grid(True, linestyle='-', alpha=0.5)

VP = ax2.boxplot(error_list_tau, positions=positions, widths=1.5, patch_artist=True,
                showmeans=False, showfliers=False,
                medianprops={"color": color, "linewidth": 0.5, "alpha":0.7},
                boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.5, "alpha":0.3},
                whiskerprops={"color": color, "linewidth": 1.5, "alpha":0.7},
                capprops={"color": color, "linewidth": 1.5, "alpha":0.7}, labels=labels)
ax2.set_ylabel('Error $\\tau$', fontsize="10" )
ax2.yaxis.grid(True, linestyle='-', alpha=0.5)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout(h_pad=0.015)

plt.show()