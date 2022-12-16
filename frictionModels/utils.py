import numpy as np


class CustomHashList(object):
    def __init__(self, num_segments):
        self.num_segments = num_segments
        self.scl = 2*self.num_segments/np.pi
        self.list = [0]*self.num_segments
        self.vel_scale = 0.01

    def get_closest_samples(self, vel, omega):
        a = np.arctan2(abs(vel/self.vel_scale), abs(omega)) * self.scl
        idx = int(a)
        if idx == self.num_segments:
            return self.list[idx-1][1], self.list[idx-1][1], 1
        return self.list[idx][0], self.list[idx][1], a

    def get_interpolation(self, vel, omega):
        v1, v2, a = self.get_closest_samples(vel, omega)
        r = a - int(a)
        return r * v2 + (1 - r) * v1

    def get_ratio_pairs(self, idx):
        r1 = np.tan(idx/self.scl)
        r2 = np.tan((idx+1)/self.scl)
        return r1, r2

    def add_to_list(self, idx, v1, v2):
        self.list[idx] = [v1, v2]

    def initialize(self, friction_model, cop):
        f1 = friction_model.step(vel_vec=vel_to_cop(-cop, {'x': 1, 'y': 0, 'tau': 0}))
        f2 = friction_model.step(vel_vec=vel_to_cop(-cop, {'x': 0, 'y': 0, 'tau': 1}))
        f_max = np.linalg.norm([f1['x'], f1['y']])
        tau_max = abs(f2['tau'])

        for i in range(self.num_segments):
            r1, r2 = self.get_ratio_pairs(i)
            w1 = np.sqrt(1/(1 + r1**2))
            v1 = self.vel_scale * np.sqrt(1 - w1**2)
            f = friction_model.step(vel_vec=vel_to_cop(-cop, {'x': v1, 'y': 0, 'tau': w1}))
            f1 = np.array([np.linalg.norm([f['x'], f['y']])/f_max, abs(f['tau'])/tau_max])

            w2 = np.sqrt(1 / (1 + r2 ** 2))
            v2 = self.vel_scale * np.sqrt(1 - w2 ** 2)
            f = friction_model.step(vel_vec=vel_to_cop(-cop, {'x': v2, 'y': 0, 'tau': w2}))
            f2 = np.array([np.linalg.norm([f['x'], f['y']]) / f_max, abs(f['tau']) / tau_max])
            self.add_to_list(i, f1, f2)


def elasto_plastic_alpha(z, z_ss, z_ba_r, v):
    """
    Calculates the alpha for elasto plastic model.
    :param z:
    :param z_ss:
    :param z_ba_r:
    :param v:
    :return:
    """
    z_norm = np.linalg.norm(z)
    z_max = np.linalg.norm(z_ss)
    z_ba = z_ba_r * z_max

    if z_norm <= z_ba:
        alpha = 0
    elif z_norm <= z_max:
        alpha = 0.5 * np.sin((z_norm - (z_max - z_ba) / 2) / (z_max - z_ba)) + 0.5
    else:
        alpha = 1

    v_norm = np.linalg.norm(v)
    if v_norm != 0 and z_norm != 0:
        v_unit = v / v_norm
        z_unit = z / z_norm
        c = v_unit.dot(z_unit)
        eps = (c + 1) / 2
        alpha = eps * alpha

    return alpha


def update_radius(full_model, fn, mu, cop):

    vel_vec_ = {'x': 0, 'y': 0, 'tau': 1}
    vel_vec = vel_to_cop(-cop, vel_vec_)

    f = full_model.step(vel_vec)

    if fn != 0:
        gamma = abs(f['tau'])/(mu*fn)
    else:
        gamma = 0

    return gamma


def update_viscus_scale(full_model, gamma, cop):
    mu_c = full_model.p['mu_c']
    mu_s = full_model.p['mu_s']
    s2 = full_model.p['s2']

    full_model.update_properties(mu_c=0, mu_s=0, s2=1)
    vel_vec_ = {'x': 0, 'y': 0, 'tau': 1}
    vel_vec = vel_to_cop(-cop, vel_vec_)
    m = full_model.fn * gamma**2
    f = full_model.step(vel_vec)
    s = abs(f['tau']/m)

    # reset parameter back
    full_model.update_properties(mu_c=mu_c, mu_s=mu_s, s2=s2)
    return np.array([1, 1, s])


def vel_to_cop(cop, vel_vec):
    u = np.array([0, 0, 1])
    w = vel_vec['tau'] * u
    pos_vex = np.zeros(3)
    pos_vex[0:2] = cop
    v_tau = np.cross(w, pos_vex)
    v_x = vel_vec['x'] + v_tau[0]
    v_y = vel_vec['y'] + v_tau[1]
    v_tau = vel_vec['tau']
    return {'x': v_x, 'y': v_y, 'tau': v_tau}
