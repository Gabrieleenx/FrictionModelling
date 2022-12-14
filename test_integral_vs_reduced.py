import numpy as np

from friction import PlanarFriction
from friction_simple import PlanarFrictionReduced
import matplotlib.pyplot as plt
from tqdm import tqdm
from pre_compute_ls import CustomHashList
dt = 1e-3
properties = {'grid_shape': (20, 20),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.3,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e5,
              's1': 2e1,
              's2': 0.4,
              'dt': dt}

properties2 = {'grid_shape': (20, 20),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e5,
              's1': 2e1,
              's2': 0,
              'dt': dt}

properties3 = {'grid_shape': (20, 20),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.3,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e5,
              's1': 2e1,
              's2': 0.4,
              'dt': dt}

def p_x_y2(M):
    return np.ones(M[1, :, :].shape)*1e3 #+ 1e3*M[1, :, :]

def vel_gen_1(t):
    vel = {'x': 0.01, 'y': 0, 'tau': 3}
    return vel

def vel_gen_2(t):
    vx = 1e-2 *t
    vy = t
    tau= 0
    vel = {'x': vx, 'y': 0, 'tau': tau}
    return vel

def vel_gen_3(t):
    if t < 0.1:
        vx = t*0.1
        tau = 0
        vy = t*0.1
    else:
        vx = 0.1*0.1
        vy = 0.1*0.1
        tau = (t-0.1)*30
    vel = {'x': vx, 'y': vy, 'tau': tau}
    return vel


def vel_gen_4(t):
    angle = np.pi * 45/180
    tau = 0
    if t < 0.1:
        vn = t

    elif t < 0.2:
        vn = 0.1 - (t - 0.1)
    elif t < 0.25:
        vn = 0
    elif t< 0.35:
        vn = 0
        tau = (t - 0.25) * 30
    elif t < 0.45:
        vn = 0
        tau = 30*0.1
    elif t < 0.75:
        vn = t - 0.45
        tau = 30 * 0.1
    elif t < 0.8:
        tau = 30 * 0.1
        vn = 0.3
    elif t < 0.9:
        tau = 30 * 0.1
        vn = 0.3 - 1.5*(t - 0.8)
    elif t < 1:
        vn = 0.15 - 1.5*(t - 0.9)
        tau = 30 * (0.1 - (t - 0.9))
    else:
        vn = 0
        tau = 0
    vx = np.cos(angle)*vn
    vy = np.sin(angle)*vn

    vel = {'x': vx, 'y': vy, 'tau': tau}
    return vel

def vel_gen_5(t):
    angle = np.pi * 45/180
    tau = 0
    if t < 0.1:
        vn = t

    elif t < 0.2:
        vn = 0.1 - (t - 0.1)
    elif t < 0.25:
        vn = 0
    elif t< 0.35:
        vn = 0
        tau = (t - 0.25) * 30
    elif t < 0.45:
        vn = 0
        tau = 30*0.1
    elif t < 0.47:
        vn = t - 0.45
        tau = 30 * 0.1
    elif t < 0.6:
        tau = 30 * 0.1
        vn = 0.02
    elif t < 0.7:
        tau = 30 * 0.1
        vn = 0.02 + (t - 0.6)
    elif t < 0.9:
        vn = 0.12 - (t-0.7)
        tau = 30 * 0.1
    elif t < 8:
        vn = -0.08
        tau = 30 * 0.1
    else:
        vn = 0
        tau = 0
    vx = np.cos(angle)*vn
    vy = np.sin(angle)*vn

    vel = {'x': vx, 'y': vy, 'tau': tau}
    return vel

def main():
    planar_lugre = PlanarFriction(properties=properties)
    planar_lugre2 = PlanarFriction(properties=properties2)

    planar_lugre_reduced = PlanarFrictionReduced(properties=properties3)
    time = 1
    n_steps = int(time/properties['dt'])
    data = np.zeros((8, n_steps))  # t, fx, fy, f_tau, vx, vy, v_tau, gamma
    data_reduced = np.zeros((8, n_steps))  # t, fx, fy, f_tau, vx, vy, v_tau, gamma

    ls_approx = CustomHashList(100)

    ls_approx.initialize(planar_lugre2, p_x_y2)

    for i in tqdm(range(n_steps)):
        data[0, i] = i * properties['dt']

        vel = vel_gen_5(i * properties['dt'])

        f = planar_lugre.step_stability_elasto_plastic(vel_vec=vel, p_x_y=p_x_y2)

        data[1, i] = f['x']
        data[2, i] = f['y']
        data[3, i] = f['tau']

        data[4, i] = vel['x']
        data[5, i] = vel['y']
        data[6, i] = vel['tau']

        f = planar_lugre_reduced.step_ellipse_stable(vel_vec=vel, p_x_y=p_x_y2, gamma=0.00764477848712988,
                                              norm_ellipse=ls_approx.get_interpolation(np.linalg.norm([vel['x'],
                                                                                                       vel['x']]),
                                                                                       abs(vel['tau'])))
        data_reduced[1, i] = f['x']
        data_reduced[2, i] = f['y']
        data_reduced[3, i] = f['tau']
        data_reduced[7, i] = f['gamma']

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 8))

    ax1.plot(data[0, :], data[1, :], alpha=0.7, label='fx')
    ax1.plot(data[0, :], data[2, :], alpha=0.7, label='fy')
    ax1.plot(data[0, :], data_reduced[1, :], '--', label='fx red')
    ax1.plot(data[0, :], data_reduced[2, :], '--', label='fy red')
    ax1.set_title('fx and fy')
    ax1.legend()

    ax2.plot(data[0, :], data[3, :], label='f tau')
    ax2.plot(data[0, :], data_reduced[3, :], '--', label='f tau red')

    ax2.set_title('torque')
    ax2.legend()

    ax3.plot(data[0, :], data[4, :], label='vx')
    ax3.plot(data[0, :], data[5, :], label='yx')
    ax3.set_title('Velocity profile')
    ax3.legend()

    ax4.plot(data[0, :], data[6, :], label='vTau')
    ax4.set_title('Velocity profile')
    ax4.legend()


    ax5.plot(data[0, :], data_reduced[7, :], label='gamma')
    ax5.set_title('gamma radius')
    ax5.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

