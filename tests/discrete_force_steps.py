import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from velocity_profiles import vel_num_cells
import surfaces.surfaces as surf
import matplotlib as mpl
from frictionModels.frictionModel import FullFrictionModel

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']


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

shape = surf.PObject(properties['grid_size'], properties['grid_shape'], surf.p_line)

planar_lugre = FullFrictionModel(properties=properties)
planar_lugre.update_p_x_y(shape)

cor = cor_max*(np.arange(n_steps)/n_steps)

f_single_data = []
f_bilinear_data = []

for i in range(n_steps):
    vel = {'x':cor[i], 'y':0.0, 'tau':1}

    f_single = planar_lugre.step_single_point(vel)
    f_bilinear = planar_lugre.step(vel)

    f_single_data.append(f_single['x'])
    f_bilinear_data.append(f_bilinear['x'])


sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})
f, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(cor, f_single_data, label='Single point', alpha=0.7)

ax.plot(cor, f_bilinear_data, label='Bilinear', alpha=0.7)
ax.set_xlabel('CoR distance from centre $[m]$')
ax.set_ylabel('Friction force $f_x$ $[N]$')
ax.legend(loc=1)
ax.set_title('Single point vs bilinear approximation')
plt.tight_layout()
plt.show()




