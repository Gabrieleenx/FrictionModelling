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




def main():
    time = 1
    grid_shape = [(10, 10), (20, 20), (30, 30), (40, 40)]
    grid_size = [2e-3, 1e-3, 2e-3 / 3, 0.5e-3]
    planar_lugre = []
    planar_lugre_reduced = []
    data = []
    data_reduced = []

    for k in range(len(grid_shape)):
        properties = {'grid_shape': grid_shape[k],  # number of grid elements in x any
                      'grid_size': grid_size[k],  # the physical size of each grid element
                      'mu_c': 1,
                      'mu_s': 1.3,
                      'v_s': 1e-3,
                      'alpha': 2,
                      's0': 1e4,
                      's1': 2e1,
                      's2': 0.4,
                      'dt': 1e-4}
        planar_lugre.append(PlanarFriction(properties=properties))
        planar_lugre_reduced.append(PlanarFrictionReduced(properties=properties))

        n_steps = int(time/properties['dt'])

        data.append(np.zeros((8, n_steps)) ) # t, fx, fy, f_tau, vx, vy, v_tau, gamma
        data_reduced.append(np.zeros((8, n_steps)))  # t, fx, fy, f_tau, vx, vy, v_tau, gamma

        for i in tqdm(range(n_steps)):
            data[k][0, i] = i * properties['dt']

            vel = vel_gen_4(i * properties['dt'])

            f = planar_lugre[k].step(vel_vec=vel, p_x_y=p_x_y2)

            data[k][1, i] = f['x']
            data[k][2, i] = f['y']
            data[k][3, i] = f['tau']
            data[k][4, i] = vel['x']
            data[k][5, i] = vel['y']
            data[k][6, i] = vel['tau']

            f = planar_lugre_reduced[k].step(vel_vec=vel, p_x_y=p_x_y2)
            data_reduced[k][1, i] = f['x']
            data_reduced[k][2, i] = f['y']
            data_reduced[k][3, i] = f['tau']
            data_reduced[k][7, i] = f['gamma']


    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 8))

    for k in range(len(grid_shape)):
        if np.max(abs(data[k][1, :])) < 100:
            ax1.plot(data[k][0, :], data[k][1, :], label='fx '+str(k))
        if np.max(abs(data[k][2, :])) < 100:
            ax1.plot(data[k][0, :], data[k][2, :], label='fy '+str(k))
        if np.max(abs(data_reduced[k][1, :])) < 100:
            ax1.plot(data[k][0, :], data_reduced[k][1, :], '--', label='fx red '+str(k))
        if np.max(abs(data_reduced[k][2, :])) < 100:
            ax1.plot(data[k][0, :], data_reduced[k][2, :], '--', label='fy red '+str(k))

        if np.max(abs(data[k][3, :])) < 100:
            ax2.plot(data[k][0, :], data[k][3, :], label='f tau '+str(k))
        if np.max(abs(data_reduced[k][3, :])) < 100:
            ax2.plot(data[k][0, :], data_reduced[k][3, :], '--', label='f tau red '+str(k))

        if np.max(abs(data_reduced[k][7, :])) < 100:
            ax5.plot(data[k][0, :], data_reduced[k][7, :], label='gamma '+str(k))

    ax1.set_title('fx and fy')
    ax1.legend()
    ax2.set_title('torque')
    ax2.legend()
    ax5.set_title('gamma radius')
    ax5.legend()

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

