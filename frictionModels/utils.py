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
    def get_closest_samples(self, vel, gamma):
        n = self.num_segments
        a1_ = np.arctan2(vel['y'], vel['x'])
        if a1_ < 0:
            a1_ = 2*np.pi + a1_
        a1 = a1_ * 2*n/np.pi


        v_xy_norm = np.linalg.norm([vel['y'], vel['x']])
        v_xy_norm = v_xy_norm/self.gamma  #  Compensate for change in gamma and the velocity relation
        # for pre-computation
        a2 = np.arctan2(v_xy_norm, abs(vel['tau'])) * 2 * n/np.pi

        r1 = int(a1)
        i1 = int(a2)
        dr = a1-r1
        di = a2-i1
        if i1 >= n:
            i1 = n-1
            di = 1
        if r1 >= 4*n:
            r1 = 4*n - 1
            dr = 1
        idx = self.calc_idx(r1, i1)
        f = self.list[idx]

        return f[0], f[1], f[2], f[3], dr, di

    def get_bilinear_interpolation(self, vel, gamma, cop=np.zeros(2)):
        """

        :param vel:
        :param gamma:
        :param cop:
        :return:
        """
        vel_cop = vel_to_cop(cop, vel)

        f1, f2, f3, f4, dr, di = self.get_closest_samples(vel_cop, gamma)

        s1 = (1-dr)*(1-di)
        s2 = dr*(1-di)
        s3 = di*(1-dr)
        s4 = dr*di

        ls_x = s1*f1['x'] + s2*f2['x'] + s3*f3['x'] + s4*f4['x']
        ls_y = s1 * f1['y'] + s2 * f2['y'] + s3 * f3['y'] + s4 * f4['y']
        ls_tau = s1 * f1['tau'] + s2 * f2['tau'] + s3 * f3['tau'] + s4 * f4['tau']
        return np.array([ls_x, ls_y, ls_tau])

    def calc_new_vel(self, vel_cop):
        # TODO
        return vel_cop

    def get_limit_surface(self, vel, gamma):
        vel_cop = vel_to_cop(self.cop, vel)
        # TODO compensate for gamma
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
            r1 = np.tan(i * np.pi/ (2*self.num_segments))

        else:
            r1 = np.tan((i+1)*np.pi / (2*self.num_segments))

        if j == 0 or j == 2:
            d = 2*np.pi * r / (4 * self.num_segments)
        else:
            d = 2 * np.pi * (r+1) / (4 * self.num_segments)

        vtau= np.sqrt(1 / (1 + r1 ** 2))
        v = self.gamma*np.sqrt(1 - vtau ** 2)
        vx = np.cos(d)*v
        vy = np.sin(d)*v
        return vx, vy, vtau

    def calc_idx(self, r, i):
        return r*self.num_segments + i

    def calc_4_points(self, friction_model, r, i, f_t_max, f_tau_max):
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
            state = False # TODO fix 
            if state:
                f[j] = value
            else:
                vx, vy, vtau = self.calc_vel(r, i, j)
                f[j] = friction_model.step(vel_vec=vel_to_cop(-self.cop, {'x': vx, 'y': vy, 'tau': vtau}))
                f[j] = normalize_force(f[j], f_t_max, f_tau_max)

        return f[0], f[1], f[2], f[3], idx

    def add_to_list(self, idx, f1, f2, f3, f4):
        self.list[idx] = [f1, f2, f3, f4]
    def initialize(self, friction_model):
        f1 = friction_model.step(vel_vec=vel_to_cop(-self.cop, {'x': 1, 'y': 0, 'tau': 0}))
        f2 = friction_model.step(vel_vec=vel_to_cop(-self.cop, {'x': 0, 'y': 0, 'tau': 1}))
        f_t_max = abs(f1['x'])
        f_tau_max = abs(f2['tau'])
        self.cop = friction_model.cop
        self.gamma = update_radius(friction_model)
        for r in range(4*self.num_segments):
            # index for rotation/direction
            for i in range(self.num_segments):
                # index for ratio between tau and f
                f1, f2, f3, f4, idx = self.calc_4_points(friction_model, r, i, f_t_max, f_tau_max)
                self.add_to_list(idx, f1, f2, f3, f4)


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

