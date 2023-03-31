import numpy as np


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

        if  vel['tau'] < 0:
            if a1_ < np.pi:
                a1_ += np.pi
            else:
                a1_ += -np.pi

        if a1_ < 0:
            a1_ = 2*np.pi + a1_
        a1 = a1_ * 2*n/np.pi


        v_xy_norm = np.linalg.norm([vel['y'], vel['x']])
        v_xy_norm = gamma*v_xy_norm/(self.gamma**2)  #  Compensate for change in gamma and the velocity relation
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


    def calc_new_vel(self, vel_cop, gamma):
        ls_0 = self.get_bilinear_interpolation(vel={'x': 0, 'y': 0, 'tau': 1}, gamma=gamma)
        ax = np.arctan2(ls_0[2], ls_0[0]) + np.pi/2
        ay = np.arctan2(ls_0[2], ls_0[1]) + np.pi/2

        iter_ = 5
        rx = ax
        ry = ay


        for i in range(iter_):
            vx = gamma * np.sin(rx)
            vy = gamma * np.sin(ry)

            ls_ = self.get_bilinear_interpolation(vel={'x': vx, 'y': vy, 'tau': 1}, gamma=gamma)
            bx = np.arctan2(ls_[2], ls_[0]) + np.pi / 2 + ax
            by = np.arctan2(ls_[2], ls_[1]) + np.pi / 2 + ay

            if ax != 0:
                rx = rx*bx/ax
            if ay != 0:
                ry = ry*by/ay

        vx = gamma * np.sin(rx)
        vy = gamma * np.sin(ry)

        p_x = -vy
        p_y = vx

        pos = np.array([p_x, p_y])*gamma/self.gamma
        new_vel = vel_to_cop(pos, vel_cop)

        p_xy_n = np.linalg.norm([p_x, p_y])
        r = 0
        if p_xy_n != 0:
            p_x_n = abs(p_x)/p_xy_n
            p_y_n = abs(p_y)/p_xy_n
            nn = np.linalg.norm([vel_cop['x']*p_x_n, vel_cop['y']*p_y_n])/self.gamma
            r = (2*np.arctan2(abs(vel_cop['tau']), nn))/(np.pi)

        new_vel_x = r*new_vel['x'] + (1-r)*vel_cop['x']
        new_vel_y = r * new_vel['y'] + (1 - r) * vel_cop['y']
        new_vel_tau = r * new_vel['tau'] + (1 - r) * vel_cop['tau']

        return {'x':new_vel_x, 'y':new_vel_y, 'tau':new_vel_tau}


    def get_if_calc(self, r, i, j):
        state = False
        value = None
        if j == 0:
            if i > 0:
                idx = self.calc_idx(r, i-1)
                value = self.list[idx][2]
                return True, value
            elif r > 0:
                idx = self.calc_idx(r - 1, i)
                value = self.list[idx][1]
                return True, value
        elif j == 1:
            if i > 0:
                idx = self.calc_idx(r, i-1)
                value = self.list[idx][3]
                return True, value
        elif j == 2:
            if r > 0:
                idx = self.calc_idx(r - 1, i)
                value = self.list[idx][3]
                return True, value

        return state, value


    def calc_vel(self, r, i, j):
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
            if state:
                f[j] = value
            else:

                vx, vy, vtau = self.calc_vel(r, i, j)

                f_ = friction_model.step(vel_vec=vel_to_cop(-self.cop, {'x': vx, 'y': vy, 'tau': vtau}))
                f_ = friction_model.force_at_cop
                f[j] = normalize_force(f_, f_t_max, f_tau_max)


        return f[0], f[1], f[2], f[3], idx

    def add_to_list(self, idx, f1, f2, f3, f4):
        self.list[idx] = [f1, f2, f3, f4]
    def initialize(self, friction_model):
        self.cop = friction_model.cop
        _ = friction_model.step(vel_vec=vel_to_cop(-self.cop, {'x': 1, 'y': 0, 'tau': 0}))
        f1 = friction_model.force_at_cop
        _ = friction_model.step(vel_vec=vel_to_cop(-self.cop, {'x': 0, 'y': 0, 'tau': 1}))
        f2 = friction_model.force_at_cop

        f_t_max = abs(f1['x'])
        f_tau_max = abs(f2['tau'])
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
    f = full_model.force_at_cop
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

