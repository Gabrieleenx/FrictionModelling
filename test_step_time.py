import numpy as np

from friction import PlanarFriction
from friction_simple import PlanarFrictionReduced
import matplotlib.pyplot as plt
from tqdm import tqdm



def p_x_y2(M):
    return np.ones(M[1, :, :].shape)*1e3 #+ 1e3*M[1, :, :]

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
    elif t < 8:
        vn = 0.12
        tau = 30 * 0.1
    else:
        vn = 0
        tau = 0
    vx = np.cos(angle)*vn
    vy = np.sin(angle)*vn

    vel = {'x': vx, 'y': vy, 'tau': tau}
    return vel

def main():
    time = 1
    dt_list = [1e-2, 1e-3, 1e-4]

    planar_lugre = []
    planar_lugre_reduced = []
    data = []
    data_reduced = []

    for k in range(len(dt_list)):
        properties = {'grid_shape': (20, 20),  # number of grid elements in x any
                      'grid_size': 1e-3,  # the physical size of each grid element
                      'mu_c': 1,
                      'mu_s': 1.3,
                      'v_s': 1e-3,
                      'alpha': 2,
                      's0': 1e4,
                      's1': 2e1,
                      's2': 0.4,
                      'dt': dt_list[k]}
        planar_lugre = PlanarFriction(properties=properties)
        planar_lugre_stable = PlanarFriction(properties=properties)

        planar_lugre_reduced = PlanarFrictionReduced(properties=properties)

        n_steps = int(time/properties['dt'])

        data.append(np.zeros((8, n_steps)) ) # t, fx, fy, f_tau, vx, vy, v_tau, gamma
        data_reduced.append(np.zeros((8, n_steps)))  # t, fx, fy, f_tau, vx, vy, v_tau, gamma

        for i in tqdm(range(n_steps)):
            data[k][0, i] = i * properties['dt']

            vel = vel_gen_4(i * properties['dt'])

            f = planar_lugre.step(vel_vec=vel, p_x_y=p_x_y2)

            data[k][1, i] = f['x']
            data[k][2, i] = f['y']
            data[k][3, i] = f['tau']
            data[k][4, i] = vel['x']
            data[k][5, i] = vel['y']
            data[k][6, i] = vel['tau']

            f = planar_lugre_stable.step_stability(vel_vec=vel, p_x_y=p_x_y2)
            data_reduced[k][1, i] = f['x']
            data_reduced[k][2, i] = f['y']
            data_reduced[k][3, i] = f['tau']


    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))

    for k in range(len(dt_list)):
        if np.max(abs(data[k][1, :])) < 100:
            ax1.plot(data[k][0, :], data[k][1, :], label='fx dt = '+str(dt_list[k]))
        if np.max(abs(data[k][2, :])) < 100:
            ax1.plot(data[k][0, :], data[k][2, :], label='fy dt = '+str(dt_list[k]))

        if np.max(abs(data_reduced[k][1, :])) < 100:
            ax1.plot(data[k][0, :], data_reduced[k][1, :], '--', label='fx stab dt = '+str(dt_list[k]))
        if np.max(abs(data_reduced[k][2, :])) < 100:
            ax1.plot(data[k][0, :], data_reduced[k][2, :], '--', label='fy stab dt = '+str(dt_list[k]))

        if np.max(abs(data[k][3, :])) < 100:
            ax2.plot(data[k][0, :], data[k][3, :], label='f tau dt = '+str(dt_list[k]))



        if np.max(abs(data_reduced[k][3, :])) < 100:
            ax2.plot(data[k][0, :], data_reduced[k][3, :], '--', label='f tau stab dt = '+str(dt_list[k]))

    ax1.set_title('fx and fy')
    ax1.legend()
    ax2.set_title('torque')
    ax2.legend()

    ax3.plot(data[0][0, :], data[0][4, :], label='vx')
    ax3.plot(data[0][0, :], data[0][5, :], label='yx')
    ax3.set_title('Velocity profile')
    ax3.legend()

    ax4.plot(data[0][0, :], data[0][6, :], label='vTau')
    ax4.set_title('Velocity profile')
    ax4.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

