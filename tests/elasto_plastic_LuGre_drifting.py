import numpy as np
import seaborn as sns
from tqdm import tqdm
from frictionModels.frictionModel import LuGre1D
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
data = {}

data['time'] = []
#data['model'] = []
#data['displacement'] = []
data['LuGre'] = []
data['elasto'] = []
data['force'] = []
data['forceMax'] = []

sim_time = 2
dt = 1e-5
n_steps = int(sim_time/dt)
freq = 10
base_f = 4
mass = 1
every_n_sample  = 100
friction = []
type_ = "tangential"  # "normal", "tangential"
properties = {
              'mu_c': 1,
              'mu_s': 1.2,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e6,
              's1': 8e1,
              's2': 0.2,
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
            if type_ == "tangential":
                f_app = base_f + 1*np.sin(2*np.pi*freq*t)
            if type_ == "normal":
                friction_model.set_fn(mass*9.81 + 1*np.sin(2*np.pi*freq*t))

        dz, f_fric = friction_model.ode_step(t, [z], [dx])
        z += dz[0]*dt
        ddx = (f_app + f_fric['x'])/mass
        dx += ddx*dt
        x += dx*dt - ddx*dt*dt/2

        if i % every_n_sample == 0:
            if model == "LuGre":
                data['time'].append(t)
                data['LuGre'].append(x)
                data['force'].append(f_app)
                data['forceMax'].append(properties['mu_s']*mass*9.81)
            else:
                data['elasto'].append(x)
            #data['model'].append(model)
            #data['displacement'].append(x)




sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})

sns.set_theme("paper", "ticks", font_scale=1.3, rc={"lines.linewidth": 2})
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))
sns.despine(f)
ax1.plot(data['time'], data['LuGre'], label="LuGre", alpha=0.7)
ax1.plot(data['time'], data['elasto'], label="Elasto-plastic", alpha=0.7)

ax1.legend(loc=1)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Position [m]')
ax1.set_title('Position')

ax2.plot(data['time'], data['force'], label="Applied force", alpha=0.7)
ax2.plot(data['time'], data['forceMax'], '--', label="Breakaway force", alpha=0.7)

ax2.legend(loc=1)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('Force [N]')
ax2.set_title('Tangential force')

"""
# Create a visualization
sns.relplot(
    data=data,
    x="time", y="displacement", hue="model", kind="line", height=4, aspect=1.5)
"""
plt.tight_layout()
plt.show()