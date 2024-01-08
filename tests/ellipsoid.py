"""
Plots the ellipse of the limit surface and compared different contact surfaces.
"""
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import surfaces.surfaces as surf
import matplotlib as mpl
from frictionModels.frictionModel import DistributedFrictionModel
from frictionModels.utils import vel_to_cop, update_radius

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

contact_size = 0.02
n_cells = 101
cor_max = 0.015
n_steps = 300

lin_vel = 0.011
ang_vel = 1
shapes = {'Circle':[surf.p_circle, 0], 'Square':[surf.p_square, 0], 'Line $v_x$':[surf.p_line, 0], 'Line $v_y$':[surf.p_line, np.pi/2]}

properties = {'grid_shape': (n_cells, n_cells),  # number of grid elements in x any
              'grid_size': contact_size/n_cells,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1,
              'v_s': 1e-3,
              'alpha': 2, # called gamma in paper
              's0': 1e6,
              's1': 8e2,
              's2': 0,
              'dt': 1e-4,
              'z_ba_ratio': 0.9,
              'stability': False,
              'elasto_plastic': False,
              'steady_state': True}

f, ax = plt.subplots(1, 1, figsize=(6, 4))

markers = [int(n_steps/4), int(n_steps/2), int(3*n_steps/4)]
shape = surf.PObject(properties['grid_size'], properties['grid_shape'], surf.p_circle)
planar_lugre = DistributedFrictionModel(properties=properties)
planar_lugre.update_p_x_y(shape)
ra = update_radius(planar_lugre)
data = np.zeros((2, n_steps))
datav = np.zeros((2, n_steps))

for i in tqdm(range(n_steps)):
    ratio = (np.pi / (2 * (n_steps - 1))) * i
    v1 = np.cos(ratio)
    v2 = np.sin(ratio)


    vx = np.cos(ratio) * lin_vel
    vy = np.cos(ratio) * 0
    vt = np.sin(ratio) * ang_vel
    datav[0, i] = vx
    datav[1, i] = vt

    v_ = np.array([vx, vy, vt])
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, pow(ra, 2)]])
    ff = A.dot(v_) / np.sqrt(v_.T.dot(A).dot(v_))

    data[0, i] = ff[0]
    data[1, i] = ff[2]

print(datav[0,markers[0]],datav[1,markers[0]])
print(datav[0,markers[1]],datav[1,markers[1]])
print(datav[0,markers[2]],datav[1,markers[2]])

max_save = np.max(abs(data), axis=1)
max_save[max_save==0] = 1
data = abs(data).T / max_save
data = data.T

ax.plot(data[0,:], data[1,:], label='Ellipse', alpha=0.7)
ax.plot(data[0, markers[0]], data[1, markers[0]], 'o', alpha=0.7, markersize=7, color='indianred')
ax.plot(data[0, markers[1]], data[1, markers[1]], 's', alpha=0.7, markersize=7, color='mediumseagreen')
ax.plot(data[0, markers[2]], data[1, markers[2]], 'v', alpha=0.7, markersize=7, color='cornflowerblue')


for k, (key, value) in enumerate(shapes.items()):
    direction = value[1]

    lin_vel_x = np.cos(direction) * lin_vel
    lin_vel_y = np.sin(direction) * lin_vel

    rot_z = np.array([[np.cos(-direction), -np.sin(-direction)],
                      [np.sin(-direction), np.cos(-direction)]])

    data = np.zeros((4, n_steps))
    shape = surf.PObject(properties['grid_size'], properties['grid_shape'], value[0])

    planar_lugre = DistributedFrictionModel(properties=properties)
    planar_lugre.update_p_x_y(shape)
    cop = planar_lugre.cop
    for i in tqdm(range(n_steps)):
        ratio = (np.pi / (2*(n_steps-1))) * i

        vx = np.cos(ratio) * lin_vel_x
        vy = np.cos(ratio) * lin_vel_y
        vt = np.sin(ratio) * ang_vel

        f = planar_lugre.step(vel_vec=vel_to_cop(-cop, vel_vec={'x': vx, 'y': vy, 'tau': vt}))
        f = planar_lugre.force_at_cop

        data[0, i] = f['x']
        data[1, i] = f['y']
        data[2, i] = f['tau']
        data[3, i] = rot_z.dot(np.array([f['x'], f['y']]).T)[0]

    max_save = np.max(abs(data), axis=1)
    max_save[max_save==0] = 1
    data = abs(data).T / max_save
    data = data.T
    ax.plot(data[3,:], data[2,:], label=key, alpha=0.7)
    ax.plot(data[3, markers[0]], data[2, markers[0]], 'o', alpha=0.9, markersize=7, color='indianred')
    ax.plot(data[3, markers[1]], data[2, markers[1]], 's', alpha=0.9, markersize=7, color='mediumseagreen')
    ax.plot(data[3, markers[2]], data[2, markers[2]], 'v', alpha=0.9, markersize=7, color='cornflowerblue')


ax.set_xlabel('$||\mathbf{f}_{t}||_2/||\mathbf{f}_{t\max}||_2$', fontsize="12" )
ax.set_ylabel('$\\tau / \\tau_{\max}$', fontsize="12" )
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., labelspacing = 0.1, fontsize="12")
ax.axis('equal')


ratio = (np.pi / (2 * (n_steps - 1))) * markers[0]
vx = np.cos(ratio) * lin_vel
vt = np.sin(ratio) * ang_vel
print(vx, vt)
ax.annotate('[$||\mathbf{v}_t||_2 = 0.0042$, $\omega = 0.93$]', xy=(0.55, 0.815), xytext=(-0.1, 0.6), fontsize=12,
            arrowprops=dict(facecolor='cornflowerblue', shrink=0.001, width=3, headwidth=8, alpha=0.6))

ax.annotate('[$||\mathbf{v}_t||_2 = 0.0078$, $\omega = 0.71$]', xy=(0.832, 0.512), xytext=(-0.1, 0.4), fontsize=12,
            arrowprops=dict(facecolor='mediumseagreen', shrink=0.001, width=3, headwidth=8, alpha=0.6))

ax.annotate('[$||\mathbf{v}_t||_2 = 0.010$, $\omega = 0.38$]', xy=(0.96, 0.24), xytext=(-0.1, 0.2), fontsize=12,
            arrowprops=dict(facecolor='indianred', shrink=0.001, width=3, headwidth=8, alpha=0.6))
plt.tight_layout(h_pad=0.015)
plt.show()



















