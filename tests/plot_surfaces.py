import numpy as np
import seaborn as sns
from frictionModels.frictionModel import LuGre1D
import matplotlib.pyplot as plt
from surfaces.surfaces import p_square, p_line, p_circle, p_line_grad, PObject
import pandas as pd



#plt.rcParams["font.family"] = "Times New Roman"

shape = 'Circle'
grid_shape = (21, 21)
grid_size = 1e-3

shape_set = {'Square': p_square, 'Circle': p_circle, 'Line': p_line, 'LineGrad': p_line_grad}
p_obj = PObject(grid_size, grid_shape, shape_set[shape])
p, cop, fn = p_obj.get(grid_size)
p = p/(grid_size**2)
np.random.seed(0)
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1})

labels = np.round((np.arange(grid_shape[0]) - (grid_shape[0]-1)/2)*grid_size, 4)
cmap = sns.color_palette('OrRd', as_cmap=True)
print(np.max(p))
ax = sns.heatmap(np.flip(p.T, axis=0), linecolor='white', linewidths=0.5, vmin=0, vmax=1.2*np.max(p), xticklabels=labels, yticklabels=-labels, annot=False, cmap=cmap)
ax.set_title('Circle', fontsize=14)
ax.set_xlabel('x $[m]$', fontsize=12)
ax.set_ylabel('y $[m]$', fontsize=12)
cb = ax.collections[0].colorbar
cb.set_label('Pressure $[N/m^2]$', fontsize=12)
plt.tight_layout()
plt.show()