import numpy as np


class CustomHashList(object):
    def __init__(self, num_segments):
        self.num_segments = num_segments
        self.scl = 2*self.num_segments/np.pi
        self.list = [0]*self.num_segments
        self.vel_scale = 0.01
        self.f_max = 0
        self.tau_max = 0

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
        self.f_max = np.linalg.norm([f1['x'], f1['y']])
        self.tau_max = abs(f2['tau'])

        for i in range(self.num_segments):
            r1, r2 = self.get_ratio_pairs(i)
            w1 = np.sqrt(1/(1 + r1**2))
            v1 = self.vel_scale * np.sqrt(1 - w1**2)
            f = friction_model.step(vel_vec=vel_to_cop(-cop, {'x': v1, 'y': 0, 'tau': w1}))
            f1 = np.array([np.linalg.norm([f['x'], f['y']])/self.f_max, abs(f['tau'])/self.tau_max])

            w2 = np.sqrt(1 / (1 + r2 ** 2))
            v2 = self.vel_scale * np.sqrt(1 - w2 ** 2)
            f = friction_model.step(vel_vec=vel_to_cop(-cop, {'x': v2, 'y': 0, 'tau': w2}))
            f2 = np.array([np.linalg.norm([f['x'], f['y']]) / self.f_max, abs(f['tau']) / self.tau_max])
            self.add_to_list(i, f1, f2)


class CustomHashList3D(object):
    def __init__(self, num_segments):
        """
        :param num_segments: Number of segments per quadrant
        """
        self.num_segments = num_segments
        self.list = [None]*4*self.num_segments*self.num_segments
        self.cop = np.zeros(2)
        self.gamma = 1
        self.omega_max = 1
    def get_closest_samples(self):
        pass

    def get_bilinear_interpolation(self, vel):
        """

        :param vel:
        :return:
        """
        pass

    def calc_new_vel(self, vel_cop):
        # TODO
        return vel_cop

    def get_limit_surface(self, vel, gamma):
        vel_cop = vel_to_cop(self.cop, vel)
        f = self.get_bilinear_interpolation(vel_cop)
        vel_hat_cop = self.calc_new_vel(vel_cop)
        vel_hat = vel_to_cop(-self.cop, vel_hat_cop)
        return f, vel_hat

    def get_if_calc(self, r, i, j):
        state = False
        value = None
        if j == 0:
            if i > 0:
                idx = self.calc_idx(r, i-1)
                value = self.list[idx][2]
                return True, value
            if r > 0:
                idx = self.calc_idx(r - 1, i)
                value = self.list[idx][1]
                return True, value
        elif j == 1:
            if i > 0:
                idx = self.calc_idx(r, i-1)
                value = self.list[idx][0]
                return True, value
        elif j == 2:
            if r > 0:
                idx = self.calc_idx(r - 1, i)
                value = self.list[idx][3]
                return True, value

        return state, value


    def calc_vel(self, r, i, j):
        v_max = self.gamma*self.omega_max
        if j == 0 or j == 1:
            v = v_max * i / (self.num_segments+1)
            vtau = self.omega_max * (self.num_segments + 1 - i) / (self.num_segments+1)
        else:
            v = v_max * (i + 1) / (self.num_segments + 1)
            vtau = self.omega_max * (self.num_segments - i) / (self.num_segments+1)

        if j == 0 or j == 2:
            d = 2*np.pi * r / (4 * self.num_segments+1)
        else:
            d = 2 * np.pi * (r+1) / (4 * self.num_segments + 1)

        vx = np.cos(d)*v
        vy = np.sin(d)*v
        return vx, vy, vtau

    def calc_idx(self, r, i):
        return r*self.num_segments + i

    def calc_4_points(self, friction_model, r, i):
        """
        Calculates the corners far a patch on the surface
        :param friction_model:
        :param r:
        :param i:
        :return:
        """
        f = [None]*4
        idx = self.calc_idx(r, i)
        for j in range(4):
            state, value = self.get_if_calc(r, i, j)
            if state:
                f[i] = value
            else:
                vx, vy, vtau = self.calc_vel(r, i, j)
                f[i] = friction_model.step(vel_vec=vel_to_cop(-self.cop, {'x': vx, 'y': vy, 'tau': vtau}))
        return f[0], f[1], f[2], f[3], idx

    def add_to_list(self, idx, f1, f2, f3, f4):
        self.list[idx] = [f1, f2, f3, f4]
    def initialize(self, friction_model):
        f1 = friction_model.step(vel_vec=vel_to_cop(-self.cop, {'x': 1, 'y': 0, 'tau': 0}))
        f2 = friction_model.step(vel_vec=vel_to_cop(-self.cop, {'x': 0, 'y': 0, 'tau': 1}))
        f_t_max = f1['x']
        f_tau_max = f2['tau']
        self.cop = friction_model.cop
        self.gamma = update_radius(friction_model)
        for r in range(4*self.num_segments):
            # index for rotation/direction
            for i in range(self.num_segments):
                # index for ratio between tau and f
                f1, f2, f3, f4, idx = self.calc_4_points(friction_model, r, i)
                self.add_to_list(idx,
                                 normalize_force(f1, f_t_max, f_tau_max),
                                 normalize_force(f2, f_t_max, f_tau_max),
                                 normalize_force(f3, f_t_max, f_tau_max),
                                 normalize_force(f4, f_t_max, f_tau_max))


def normalize_force(f, f_t_max, f_tau_max):
    f_norm = {'x': f['x']/f_t_max, 'y': f['y']/f_t_max, 'tau': f['tau']/f_tau_max}
    return f_norm

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
        alpha = 0.5 * np.sin(np.pi*((z_norm - (z_max + z_ba) / 2) / (z_max - z_ba))) + 0.5
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


def update_radius(full_model):
    cop = full_model.cop
    fn = full_model.fn
    mu = full_model.p['mu_c']
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

