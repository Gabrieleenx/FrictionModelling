
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import surfaces.surfaces as surf
import matplotlib as mpl
from frictionModels.frictionModel import FullFrictionModel
from frictionModels.utils import vel_to_cop

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.5, rc={"lines.linewidth": 2})



contact_size = 0.02
n_cells = 21
cor_max = 0.015
n_steps = int(1e4)

properties = {'grid_shape': (n_cells, n_cells),  # number of grid elements in x any
              'grid_size': contact_size/n_cells,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e5,
              's1': 2e1,
              's2': 0,
              'dt': 1e-4,
              'z_ba_ratio': 0.9,
              'stability': False,
              'elasto_plastic': True,
              'steady_state': True}

shape = surf.PObject(properties['grid_size'], properties['grid_shape'], surf.p_line_grad)
planar_lugre = FullFrictionModel(properties=properties)
planar_lugre.update_p_x_y(shape)

num = 20
lin_vel = 0.01
ang_vel = 1

data3d = np.zeros((3, 4 * num * (2 * num)))
direction_ = np.linspace(0, 2 * np.pi, 4 * num)

for j in range(4 * num):
    direction = direction_[j]
    lin_vel_x = np.cos(direction) * lin_vel
    lin_vel_y = np.sin(direction) * lin_vel

    cop = planar_lugre.cop
    v1 = vel_to_cop(-cop, vel_vec={'x': 0, 'y': 0, 'tau': ang_vel})
    v2 = vel_to_cop(-cop, vel_vec={'x': lin_vel_x, 'y': lin_vel_y, 'tau': 0})

    ang_vel_list = np.linspace(v1['tau'], v2['tau'], num)
    lin_vel_x_list = np.linspace(v1['x'], v2['x'], num)
    lin_vel_y_list = np.linspace(v1['y'], v2['y'], num)

    ang_vel_list = np.hstack([ang_vel_list, np.linspace(-v1['tau'], v2['tau'], num)])
    lin_vel_x_list = np.hstack([lin_vel_x_list, np.linspace(-v1['x'], v2['x'], num)])
    lin_vel_y_list = np.hstack([lin_vel_y_list, np.linspace(-v1['y'], v2['y'], num)])

    for i in range(2*num):
        f = planar_lugre.step(vel_vec={'x': lin_vel_x_list[i], 'y': lin_vel_y_list[i], 'tau': ang_vel_list[i]})
        f = planar_lugre.force_at_cop
        data3d[0, i + j * 2*num] = f['x']
        data3d[1, i + j * 2*num] = f['y']
        data3d[2, i + j * 2*num] = f['tau']

max_save3d = np.max(abs(data3d), axis=1)
data3d = data3d.T / max_save3d
data3d = data3d.T


fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(projection='3d')

ax.scatter(data3d[0,:], data3d[1,:], data3d[2,:], c=data3d[2,:], cmap='viridis')
ax.set_xlabel('$f_x/f_{x{\max}}$' )
ax.set_ylabel('$f_y/f_{y{\max}}$')
ax.set_zlabel('$\\tau /\\tau_{\max} $' )
# Disable rotation for x-axis, y-axis, and z-axis labels

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.axis('equal')
ax.view_init(elev=20, azim=69, roll=0)

# Set the limits of the axes
plt.tight_layout()

plt.show()

