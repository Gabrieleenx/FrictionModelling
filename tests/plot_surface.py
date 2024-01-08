import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from surfaces.surfaces import p_square, p_line, p_circle, p_line_grad, p_line_grad2, p_line_grad3, p_line_grad4, PObject, non_convex_1, non_convex_2, proportional_surface_circle
import pandas as pd

import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.formatter.limits'] = (-2, 3)
sns.set_theme("paper", "ticks", font_scale=1.0, rc={"lines.linewidth": 2})

shape = 'NonConvex1'
grid_shape = (21, 21)
grid_size = 1e-3

shape_set = {'Square': p_square, 'Circle': p_circle, 'Line': p_line, 'LineGrad': p_line_grad,
             'NonConvex1':non_convex_1, 'NonConvex2':non_convex_2, 'LineGrad2': p_line_grad, 'LineGrad3': p_line_grad3,
             'LineGrad4': p_line_grad4}
p_obj = PObject(grid_size, grid_shape, shape_set[shape])
#p_obj = PObject(grid_size, grid_shape, proportional_surface_circle, 2)
p, cop, fn = p_obj.get(grid_size)
p = p/(grid_size**2)
np.random.seed(0)
sns.set_theme(rc={'axes.formatter.limits': (0, 0)})
sns.set(rc={'figure.figsize':(5,4.2)})

labels = np.round((np.arange(grid_shape[0]) - (grid_shape[0]-1)/2)*grid_size*1e3, 1)
cmap = sns.color_palette('OrRd', as_cmap=True)
ax = sns.heatmap(p, linecolor='white', linewidths=0.5, vmin=0, vmax=1.2*np.max(p), xticklabels=labels, yticklabels=-labels, annot=False, cmap=cmap)
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)

#ax.set_title("Gradient line", fontsize=18)
ax.set_xlabel('$x$ $[mm]$', fontsize=22)
ax.set_ylabel('$y$ $[mm]$', fontsize=22)

cb = ax.collections[0].colorbar
cb.set_label('Pressure $[N/m^2]$', fontsize=20)
plt.tight_layout()
plt.show()