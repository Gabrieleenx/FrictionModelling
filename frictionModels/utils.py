import numpy as np


class CustomHashList3D(object):
    def __init__(self, num_segments):
        """
        This class is used for pre-computing the limit surface.
        :param num_segments: Number of segments per quadrant
        """
        self.num_segments = num_segments
        self.list = [None]*4*self.num_segments*self.num_segments
        self.cop = np.zeros(2)
        self.ra = 1

    def get_closest_samples(self, vel, ra):
        """
        Get the closest samples and location
        :param vel: {'x': vx, 'y': vy, 'tau': vtau}
        :param ra: radius
        :return: f[0], f[1], f[2], f[3], dr, di
        """
        n = self.num_segments
        a1_ = np.arctan2(vel['y'], vel['x'])

        if vel['tau'] < 0:
            if a1_ < np.pi:
                a1_ += np.pi
            else:
                a1_ += -np.pi

        if a1_ < 0:
            a1_ = 2*np.pi + a1_

        a1 = a1_ * 2*n/np.pi


        v_xy_norm = np.linalg.norm([vel['y'], vel['x']])

        v_xy_norm = v_xy_norm
        a2 = np.arctan2(v_xy_norm, ra * abs(vel['tau'])) * 2 * n/np.pi

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

    def get_bilinear_interpolation(self, vel, ra, cop=np.zeros(2)):
        """
        Calculate the bilinear interpolation
        :param vel: {'x': fx, 'y': fy, 'tau': ftau}
        :param ra: radius
        :param cop: np.array (2,)
        :return: np.array (3,)
        """
        vel_cop = vel_to_cop(cop, vel)

        f1, f2, f3, f4, dr, di = self.get_closest_samples(vel_cop, ra)

        s1 = (1-dr)*(1-di)
        s2 = dr*(1-di)
        s3 = di*(1-dr)
        s4 = dr*di

        ls_x = s1*f1['x'] + s2*f2['x'] + s3*f3['x'] + s4*f4['x']
        ls_y = s1 * f1['y'] + s2 * f2['y'] + s3 * f3['y'] + s4 * f4['y']
        ls_tau = s1 * f1['tau'] + s2 * f2['tau'] + s3 * f3['tau'] + s4 * f4['tau']

        return np.array([[ls_x], [ls_y], [ls_tau]])


    def get_if_calc(self, r, i, j):
        """
        Check if we can reuse calculations.
        :param r: index
        :param i: index
        :param j: index
        :return: bool, {'x': fx, 'y': fy, 'tau': ftau}
        """
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
        """
        Calculate velocity for a index
        :param r: index
        :param i: index
        :param j: index
        :return: vx, vy, vtau
        """
        if j == 0 or j == 1:
            r1 = np.tan(i * np.pi/ (2*self.num_segments))

        else:
            r1 = np.tan((i+1)*np.pi / (2*self.num_segments))

        if j == 0 or j == 2:
            d = 2*np.pi * r / (4 * self.num_segments)
        else:
            d = 2 * np.pi * (r+1) / (4 * self.num_segments)

        vtau= np.sqrt(1 / (1 + r1 ** 2))
        v = self.ra*np.sqrt(1 - vtau ** 2)
        vx = np.cos(d)*v
        vy = np.sin(d)*v
        return vx, vy, vtau

    def calc_idx(self, r, i):
        """
        From two dim to one dim index
        :param r: index direction on fx and fy
        :param i: index angle between torque and fx, fy
        :return: index
        """
        return r*self.num_segments + i

    def calc_4_points(self, friction_model, r, i, f_t_max, f_tau_max):
        """
        Calculates the corners far a patch on the surface
        :param friction_model: friction model class
        :param r: index
        :param i: index
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
        """
        Adds to the hash list
        :param idx: index in hash list
        :param f1: {'x': fx, 'y': fy, 'tau': ftau}
        :param f2: {'x': fx, 'y': fy, 'tau': ftau}
        :param f3: {'x': fx, 'y': fy, 'tau': ftau}
        :param f4: {'x': fx, 'y': fy, 'tau': ftau}
        :return:
        """
        self.list[idx] = [f1, f2, f3, f4]

    def initialize(self, friction_model):
        """
        Initialize with friction model class
        :param friction_model: friction model class
        :return:
        """
        self.cop = friction_model.cop
        f1_ = friction_model.step(vel_vec=vel_to_cop(-self.cop, {'x': 1, 'y': 0, 'tau': 0}))
        f1 = friction_model.force_at_cop
        f2_ = friction_model.step(vel_vec=vel_to_cop(-self.cop, {'x': 0, 'y': 0, 'tau': 1}))
        f2 = friction_model.force_at_cop

        f_t_max = abs(f1['x'])
        f_tau_max = abs(f2['tau'])
        self.ra = update_radius(friction_model)
        for r in range(4*self.num_segments):
            # index for rotation/direction
            for i in range(self.num_segments):
                # index for ratio between tau and f
                f1, f2, f3, f4, idx = self.calc_4_points(friction_model, r, i, f_t_max, f_tau_max)
                self.add_to_list(idx, f1, f2, f3, f4)


def normalize_force(f, f_t_max, f_tau_max):
    """
    normalize the force
    :param f: {'x': fx, 'y': fy, 'tau': ftau}
    :param f_t_max: max translational force flaat
    :param f_tau_max: max torque float
    :return:
    """
    f_norm = {'x': f['x']/f_t_max, 'y': f['y']/f_t_max, 'tau': f['tau']/f_tau_max}
    return f_norm

def elasto_plastic_beta(z, z_ss, z_ba_r, v):
    """
    Calculates beta for the elasto-plastic model.
    :param z: bristle deformation np.array (2,) or (3,)
    :param z_ss: steady state deflection np.array (2,) or (3,)
    :param z_ba_r: ratio between 0-1 float
    :param v: np.array (2,) or (3,)
    :return: float
    """
    z_norm = np.linalg.norm(z)
    z_max = np.linalg.norm(z_ss)
    z_ba = z_ba_r * z_max

    if z_norm <= z_ba:
        beta = 0
    elif z_norm <= z_max:
        beta = 0.5 * np.sin(np.pi*((z_norm - (z_max + z_ba) / 2) / (z_max - z_ba))) + 0.5
    else:
        beta = 1
    v_norm = np.linalg.norm(v)
    if v_norm != 0 and z_norm != 0:
        v_unit = v / v_norm
        z_unit = z / z_norm
        c = v_unit.dot(z_unit)
        eps = (c + 1) / 2
        beta = eps * beta

    return beta


def update_radius(distributed_model):
    """
    Calculates the radius for the reduced model
    :param distributed_model: python class
    :return: radius
    """
    cop = distributed_model.cop
    fn = distributed_model.fn
    mu = distributed_model.p['mu_c']
    vel_vec_ = {'x': 0, 'y': 0, 'tau': 1}
    vel_vec = vel_to_cop(-cop, vel_vec_)
    f = distributed_model.step(vel_vec)
    f = distributed_model.force_at_cop
    if fn != 0:
        ra = abs(f['tau'])/(mu*fn)
    else:
        ra = 0
    return ra


def update_viscus_scale(distributed_model, ra, cop):
    """
    Calculates scaling for viscus friction
    :param distributed_model: python class
    :param ra: radius
    :param cop: np.array([x, y])
    :return:
    """
    mu_c = distributed_model.p['mu_c']
    mu_s = distributed_model.p['mu_s']
    s2 = distributed_model.p['s2']

    distributed_model.update_properties(mu_c=0, mu_s=0, s2=1)
    vel_vec_ = {'x': 0, 'y': 0, 'tau': 1}
    vel_vec = vel_to_cop(-cop, vel_vec_)
    m = distributed_model.fn * ra ** 2
    f = distributed_model.step(vel_vec)
    s = abs(f['tau']/m)

    # reset parameter back
    distributed_model.update_properties(mu_c=mu_c, mu_s=mu_s, s2=s2)
    return np.array([[1], [1], [s]])


def vel_to_cop(cop, vel_vec):
    """
    Move velocity to cop
    :param cop: np.array([x, y])
    :param vel_vec: {'x': v_x, 'y': v_y, 'tau': v_tau}
    :return: {'x': v_x, 'y': v_y, 'tau': v_tau}
    """
    u = np.array([0, 0, 1])
    w = vel_vec['tau'] * u
    pos_vex = np.zeros(3)
    pos_vex[0:2] = cop
    v_tau = np.cross(w, pos_vex)
    v_x = vel_vec['x'] + v_tau[0]
    v_y = vel_vec['y'] + v_tau[1]
    v_tau = vel_vec['tau']
    return {'x': v_x, 'y': v_y, 'tau': v_tau}



