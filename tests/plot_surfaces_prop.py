import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from surfaces.surfaces import p_square, p_line, p_circle, p_line_grad, p_line_grad2, p_line_grad3, p_line_grad4, PObject, non_convex_1, non_convex_2, proportional_surface_circle, proportional_surface_circle2
import pandas as pd
from matplotlib.ticker import MultipleLocator

import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})
sns.set_theme(rc={'axes.formatter.limits': (0, 0)})
sns.set(rc={'figure.figsize': (5, 4.2)})

fN = [1, 1.5, 2, 2.5]

shape_set = {'Square': p_square, 'Circle': p_circle, 'Line': p_line, 'LineGrad': p_line_grad,
             'NonConvex1':non_convex_1, 'NonConvex2':non_convex_2, 'LineGrad2': p_line_grad, 'LineGrad3': p_line_grad3,
             'LineGrad4': p_line_grad4}
f, ax = plt.subplots(2, 4, figsize=(8, 4))
grid_shape = (21, 21)
grid_size = 1e-3
i = 0
for fn in fN:

    p1 = proportional_surface_circle(grid_shape, fn)
    p1 = (p1/np.sum(p1))*fn / (grid_size ** 2)

    p2 = proportional_surface_circle2(grid_shape, fn)
    p2 = (p2/np.sum(p2))*fn / (grid_size ** 2)

    np.random.seed(0)


    labels = np.round((np.arange(grid_shape[0]) - (grid_shape[0]-1)/2)*grid_size*1e3, 1)
    cmap = sns.color_palette('OrRd', as_cmap=True)
    ax_1 = sns.heatmap(p1, linecolor='white', linewidths=0.5, vmin=0, vmax=1.2*np.max(p1), xticklabels=labels,
                      yticklabels=-labels, annot=False, cmap=cmap, ax=ax[0, i])

    ax_2 = sns.heatmap(p2, linecolor='white', linewidths=0.5, vmin=0, vmax=1.2 * np.max(p2), xticklabels=labels,
                      yticklabels=-labels, annot=False, cmap=cmap, ax=ax[1, i])

    ax_1.get_xaxis().set_visible(False)
    if i != 0:
        ax_2.get_yaxis().set_visible(False)
        ax_1.get_yaxis().set_visible(False)

    ax_1.tick_params(axis='y', labelsize=7)
    ax_1.tick_params(axis='x', labelsize=7)
    ax_2.tick_params(axis='y', labelsize=7)
    ax_2.tick_params(axis='x', labelsize=7)
    for k, label in enumerate(ax_1.yaxis.get_ticklabels()):
        if k % 2 != 0:
            label.set_visible(False)

    for k, label in enumerate(ax_1.xaxis.get_ticklabels()):
        if k % 2 != 0:
            label.set_visible(False)
    for k, label in enumerate(ax_2.yaxis.get_ticklabels()):
        if k % 2 != 0:
            label.set_visible(False)

    for k, label in enumerate(ax_2.xaxis.get_ticklabels()):
        if k % 2 != 0:
            label.set_visible(False)
    ax_1.set_xlabel('$x$ $[mm]$', fontsize=11)
    ax_1.set_ylabel('$y$ $[mm]$', fontsize=11)
    ax_1.set_title('$f_N = $' + str(fn), fontsize=9)
    cb = ax_1.collections[0].colorbar
    cb.ax.tick_params(labelsize=7)
    if i == 3:
        cb.set_label('Pressure $[N/m^2]$', fontsize=11)


    ax_2.set_xlabel('$x$ $[mm]$', fontsize=11)
    ax_2.set_ylabel('$y$ $[mm]$', fontsize=11)
    ax_2.set_title(fn, fontsize=9)
    cb = ax_2.collections[0].colorbar
    cb.ax.tick_params(labelsize=7)
    if i == 3:
        cb.set_label('Pressure $[N/m^2]$', fontsize=11)

    i += 1
plt.tight_layout()
plt.show()