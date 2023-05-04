import numpy as np
import seaborn as sns
from tqdm import tqdm
from frictionModels.frictionModel import LuGre1D
import matplotlib.pyplot as plt

data = {}

data['time'] = []
data['model'] = []
data['displacement'] = []
sim_time = 2
dt = 1e-5
n_steps = int(sim_time/dt)
freq = 10
base_f = 0.4
mass = 1
every_n_sample  = 100
friction = []
properties = {
              'mu_c': 1,
              'mu_s': 1.3,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e5,
              's1': 2e1,
              's2': 0.4,
              'dt': 1e-4,
              'z_ba_ratio': 0.9,
              'elasto_plastic': True}

for model in ['LuGre', 'Elasto-plasitc']:
    x = 0
    dx = 0
    z = 0
    if model == 'LuGre':
        properties['elasto_plastic'] = False
    else:
        properties['elasto_plastic'] = True

    friction_model = LuGre1D(properties=properties, fn=mass*9.81)
    for i in range(n_steps):
        t = i * dt
        if t < 0.2:
            f_app = base_f * t/0.2
        elif t < 0.5:
            f_app = base_f
        else:
            f_app = base_f + 0.1*np.sin(2*np.pi*freq*t)

        dz, f_fric = friction_model.ode_step(t, [z], [dx])
        z += dz[0]*dt
        ddx = (f_app + f_fric['x'])/mass
        dx += ddx*dt
        x += dx*dt - ddx*dt*dt/2

        if i % every_n_sample == 0:
            data['time'].append(t)
            data['model'].append(model)
            data['displacement'].append(x)




sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})


# Create a visualization
sns.relplot(
    data=data,
    x="time", y="displacement", hue="model", kind="line", height=4, aspect=1.5)

plt.show()