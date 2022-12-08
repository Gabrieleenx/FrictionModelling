import numpy as np
import matplotlib.pyplot as plt




def torque_ss(vel_vec, gamma, f_n, p):
    vel_at_cop = np.array([vel_vec['x'], vel_vec['y'], vel_vec['tau']])
    w = vel_vec['tau']
    mc = p['mu_c']
    ms = p['mu_s']
    vs = p['v_s']
    a = p['alpha']
    vel_at_cop[2] = vel_at_cop[2] * gamma

    v_norm = np.linalg.norm(vel_at_cop)

    g = mc + (ms - mc) * np.exp(- (v_norm / vs) ** a)

    f_tau_ss = -(g / v_norm + p['s2']) * f_n * vel_at_cop[2] * gamma

    Q1 = gamma * w ** 2 * g / (v_norm ** 3)
    Q2 = (g - mc) * a * gamma * w ** 2 * (v_norm / vs) ** (a - 1)
    Q3 =p['s2'] + g / v_norm
    df_dgamma = vel_at_cop[2] * f_n * (gamma * (Q1 - Q2) - 2 * Q3)
    return f_tau_ss, df_dgamma


m_s_list = [1, 1.5, 2, 3, 4, 5, 10]
vel_vec = {'x': 0.001, 'y': 0, 'tau': 0.15}
step_size = 1e-4
end_ = 0.02
n_steps = int(end_/step_size)
f_n = 1
f, (ax1) = plt.subplots(1, 1, figsize=(12, 8))

for i in range(len(m_s_list)):
    data = np.zeros((2, n_steps))
    p = {'grid_shape': (20, 20),  # number of grid elements in x any
         'grid_size': 1e-3,  # the physical size of each grid element
         'mu_c': 1,
         'mu_s': m_s_list[i],
         'v_s': 1e-3,
         'alpha': 2,
         's0': 1e5,
         's1': 2e1,
         's2': 0.4,
         'dt': 1e-4}

    for k in range(n_steps):
        gamma = k*step_size
        f_t, df_t = torque_ss(vel_vec, gamma, f_n, p)
        data[0, k] = gamma
        data[1, k] = f_t


    ax1.plot(data[0, :], data[1, :], alpha=0.7, label='f_tau_ss '+ str(m_s_list[i]))

ax1.set_title('Gamma and mu_s to mu_c ratio w =0.15, vx=0.001')
ax1.legend()


plt.tight_layout()
plt.xlabel('gamma')
plt.ylabel('torque ss')
plt.show()
