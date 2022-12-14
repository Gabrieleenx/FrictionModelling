import numpy
import numpy as np
from typing import Dict, Callable


class PlanarFrictionReduced(object):
    def __init__(self, properties: Dict[str, any]):
        self.cop = np.zeros(2)
        self.p = properties

        self.velocity_grid = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))

        self.normal_force_grid = np.zeros((self.p['grid_shape'][0], self.p['grid_shape'][1]))

        self.x_pos_vec = (np.arange(self.p['grid_shape'][0]) + 0.5 - self.p['grid_shape'][0]/2) * self.p['grid_size']
        self.y_pos_vec = (np.arange(self.p['grid_shape'][1]) + 0.5 - self.p['grid_shape'][1]/2) * self.p['grid_size']

        self.pos_matrix = np.zeros((self.p['grid_shape'][0], self.p['grid_shape'][1], 3))
        for i_x in range(self.p['grid_shape'][0]):
            for i_y in range(self.p['grid_shape'][1]):
                self.pos_matrix[i_x, i_y, 0] = self.x_pos_vec[i_x]
                self.pos_matrix[i_x, i_y, 1] = self.y_pos_vec[i_y]
        self.pos_matrix_2d = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))
        self.pos_matrix_2d[0, :, :] = self.pos_matrix[:, :, 0]
        self.pos_matrix_2d[1, :, :] = self.pos_matrix[:, :, 1]
        z = np.zeros(3)  # bristles
        f = np.zeros(3)  # tangential force at each grid cell
        self.lugre = {'z': z, 'f': f}
        self.gamma = 0.05
        self.fn = 0

    def step(self, vel_vec: Dict[str, float], p_x_y: Callable[[float, float], float]) -> Dict[str, float]:
        """
        k
        :param vel_vec: velocity vector in the center of the censor
        :param p_x_y: function that takes x, y and returns p
        :return: force in x, y and moment expressed in the center of the sensor.
        """

        self.update_cop_and_force_grid(p_x_y)
        vel_at_cop = vel_vec  # ToDO!
        gamma = self.update_radius(vel_at_cop)

        self.update_lugre(vel_at_cop)

        force_at_cop = {'x': self.lugre['f'][0], 'y': self.lugre['f'][1], 'tau': self.lugre['f'][2]}

        force = self.move_force_to_center(force_at_cop)

        return force

    def step_ellipse(self, vel_vec: Dict[str, float], p_x_y: Callable[[float, float], float],
                                       gamma: float, norm_ellipse: np.array) -> Dict[str, float]:
        """
        k
        :param norm_ellipse:
        :param gamma:
        :param vel_vec: velocity vector in the center of the censor
        :param p_x_y: function that takes x, y and returns p
        :return: force in x, y and moment expressed in the center of the sensor.
        """

        self.update_cop_and_force_grid(p_x_y)
        vel_at_cop = vel_vec
        self.gamma = gamma
        self.update_lugre_ellipse(vel_at_cop, norm_ellipse)

        force_at_cop = {'x': self.lugre['f'][0], 'y': self.lugre['f'][1], 'tau': self.lugre['f'][2]}

        force = self.move_force_to_center(force_at_cop)

        return force

    def step_ellipse_stable(self, vel_vec: Dict[str, float], p_x_y: Callable[[float, float], float],
                                       gamma: float, norm_ellipse: np.array) -> Dict[str, float]:
        """
        k
        :param norm_ellipse:
        :param gamma:
        :param vel_vec: velocity vector in the center of the censor
        :param p_x_y: function that takes x, y and returns p
        :return: force in x, y and moment expressed in the center of the sensor.
        """

        self.update_cop_and_force_grid(p_x_y)
        vel_at_cop = vel_vec
        self.gamma = gamma
        self.update_lugre_ellipse_stable(vel_at_cop, norm_ellipse)

        force_at_cop = {'x': self.lugre['f'][0], 'y': self.lugre['f'][1], 'tau': self.lugre['f'][2]}

        force = self.move_force_to_center(force_at_cop)

        return force

    def step_ellipse_stable_elasto_plastic(self, vel_vec: Dict[str, float], p_x_y: Callable[[float, float], float],
                                       gamma: float, norm_ellipse: np.array) -> Dict[str, float]:
        """
        k
        :param norm_ellipse:
        :param gamma:
        :param vel_vec: velocity vector in the center of the censor
        :param p_x_y: function that takes x, y and returns p
        :return: force in x, y and moment expressed in the center of the sensor.
        """

        self.update_cop_and_force_grid(p_x_y)
        vel_at_cop = vel_vec
        self.gamma = gamma
        self.update_lugre_ellipse_stable_elasto_plastic(vel_at_cop, norm_ellipse)

        force_at_cop = {'x': self.lugre['f'][0], 'y': self.lugre['f'][1], 'tau': self.lugre['f'][2]}

        force = self.move_force_to_center(force_at_cop)

        return force

    def steady_state(self, vel_vec: Dict[str, float], p_x_y: Callable[[float, float], float]) -> Dict[str, float]:
        """
        k
        :param vel_vec: velocity vector in the center of the censor
        :param p_x_y: function that takes x, y and returns p
        :return: force in x, y and moment expressed in the center of the sensor.
        """

        self.update_cop_and_force_grid(p_x_y)
        vel_at_cop = vel_vec
        gamma = self.update_radius(vel_at_cop)

        self.update_lugre_ss(vel_at_cop)

        force_at_cop = {'x': self.lugre['f'][0], 'y': self.lugre['f'][1], 'tau': self.lugre['f'][2]}

        force = self.move_force_to_center(force_at_cop)

        return force, self.gamma

    def steady_state_gamma(self, vel_vec: Dict[str, float], p_x_y: Callable[[float, float], float], gamma: float) -> Dict[str, float]:
        """
        k
        :param gamma:
        :param vel_vec: velocity vector in the center of the censor
        :param p_x_y: function that takes x, y and returns p
        :return: force in x, y and moment expressed in the center of the sensor.
        """

        self.update_cop_and_force_grid(p_x_y)
        vel_at_cop = vel_vec
        self.gamma = gamma
        self.update_lugre_ss(vel_at_cop)

        force_at_cop = {'x': self.lugre['f'][0], 'y': self.lugre['f'][1], 'tau': self.lugre['f'][2]}

        force = self.move_force_to_center(force_at_cop)

        return force

    def steady_state_gamma_ellipse(self, vel_vec: Dict[str, float], p_x_y: Callable[[float, float], float],
                                   gamma: float, norm_ellipse: np.array) -> Dict[str, float]:
        """
        k
        :param norm_ellipse:
        :param gamma:
        :param vel_vec: velocity vector in the center of the censor
        :param p_x_y: function that takes x, y and returns p
        :return: force in x, y and moment expressed in the center of the sensor.
        """

        self.update_cop_and_force_grid(p_x_y)
        vel_at_cop = vel_vec
        self.gamma = gamma
        self.update_lugre_ss_ellipse(vel_at_cop, norm_ellipse)

        force_at_cop = {'x': self.lugre['f'][0], 'y': self.lugre['f'][1], 'tau': self.lugre['f'][2]}

        force = self.move_force_to_center(force_at_cop)

        return force

    def update_cop_and_force_grid(self, p_x_y):
        area = self.p['grid_size'] ** 2
        self.normal_force_grid = p_x_y(self.pos_matrix_2d)
        self.normal_force_grid = self.normal_force_grid * area
        self.f_n = np.sum(self.normal_force_grid)
        self.cop[0] = np.sum(self.x_pos_vec.dot(self.normal_force_grid)) / self.f_n
        self.cop[1] = np.sum(self.y_pos_vec.dot(self.normal_force_grid.T)) / self.f_n

    def update_velocity_grid(self, vel_vec):
        u = np.array([0, 0, 1])
        w = vel_vec['tau'] * u
        v_tau = np.cross(w, self.pos_matrix)
        self.velocity_grid[0, :, :] = v_tau[:, :, 0] + vel_vec['x']
        self.velocity_grid[1, :, :] = v_tau[:, :, 1] + vel_vec['y']

    def update_radius(self, vel_at_cop):
        if abs(vel_at_cop['tau']) > 1e-5:
            vel_vec_ = vel_at_cop
        else:
            vel_vec_ = {'x': 0, 'y': 0, 'tau': 1e-5}

        self.update_velocity_grid(vel_vec_)
        f_tau_ss =  self.calculate_ss_torque()

        gamma = self.gamma
        gamma_old = 0
        while abs(gamma - gamma_old) > 1e-6:
            gamma_old = gamma
            f_tau_ss_rim, df_dgamma = self.torque_ss(vel_vec_, gamma)
            gamma = gamma - (f_tau_ss_rim - f_tau_ss) / df_dgamma
        self.gamma = gamma

        return self.gamma

    def torque_ss(self, vel_vec, gamma):
        vel_at_cop = np.array([vel_vec['x'], vel_vec['y'], vel_vec['tau']])
        w = vel_vec['tau']
        mc = self.p['mu_c']
        ms = self.p['mu_s']
        vs = self.p['v_s']
        a = self.p['alpha']
        vel_at_cop[2] = vel_at_cop[2] * gamma

        v_norm = np.linalg.norm(vel_at_cop)

        g = mc + (ms - mc) * np.exp(- (v_norm / vs) ** a)

        f_tau_ss = -(g/v_norm + self.p['s2']) * self.f_n * vel_at_cop[2] * gamma

        Q1 = gamma * w**2 * g / (v_norm**3)
        Q2 = (g - mc) * a * gamma * w**2 * (v_norm/vs)**(a-1)
        Q3 = self.p['s2'] + g/v_norm
        df_dgamma = vel_at_cop[2] * self.f_n*(gamma*(Q1 - Q2) - 2*Q3)
        return f_tau_ss, df_dgamma

    def calculate_ss_torque(self):
        v_norm = np.linalg.norm(self.velocity_grid, axis=0)
        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * \
            np.exp(- (v_norm / self.p['v_s']) ** self.p['alpha'])
        v_norm[v_norm==0] = 1
        l_ss = ((self.velocity_grid/v_norm)*g + self.p['s2'] * self.velocity_grid) * self.normal_force_grid
        tau = np.cross(self.pos_matrix_2d, l_ss, axis=0)
        return -np.sum(tau)

    def update_lugre(self, vel_at_cop):
        vel_at_cop_ = np.array([vel_at_cop['x'], vel_at_cop['y'], vel_at_cop['tau']*self.gamma])

        v_norm = np.linalg.norm(vel_at_cop_)

        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * np.exp(- (v_norm/self.p['v_s'])**self.p['alpha'])

        dz = vel_at_cop_ - self.lugre['z'] * (self.p['s0'] * (v_norm / g))

        self.lugre['f'] = -(self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz + self.p['s2'] * vel_at_cop_) * self.f_n
        self.lugre['f'][2] = self.lugre['f'][2]*self.gamma
        self.lugre['z'] += dz * self.p['dt']

    def update_lugre_ss(self, vel_at_cop):
        vel_at_cop_ = np.array([vel_at_cop['x'], vel_at_cop['y'], vel_at_cop['tau']*self.gamma])

        v_norm = np.linalg.norm(vel_at_cop_)
        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * np.exp(- (v_norm/self.p['v_s'])**self.p['alpha'])
        if v_norm == 0:
            v_norm = 1
        self.lugre['z'] = vel_at_cop_ / (self.p['s0'] * (v_norm / g))
        scaling = np.array([1, 1, (9/8)])

        self.lugre['f'] = -(self.p['s0'] * self.lugre['z']+ scaling*self.p['s2'] * vel_at_cop_) * self.f_n
        self.lugre['f'][2] = self.lugre['f'][2]*self.gamma

    def update_lugre_ss_ellipse(self, vel_at_cop, ellipse):
        vel_at_cop_ = np.array([vel_at_cop['x'], vel_at_cop['y'], vel_at_cop['tau']*self.gamma])
        v_norm = np.linalg.norm(vel_at_cop_)

        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * np.exp(- (v_norm/self.p['v_s'])**self.p['alpha'])

        if v_norm == 0:
            beta = np.ones(3)
        else:
            v_norm_t = np.linalg.norm(vel_at_cop_[0:2])
            ellipse_norm = 1 # np.linalg.norm(ellipse)
            if abs(ellipse[1]) > 1e-10 and vel_at_cop_[2] != 0:
                beta_tau = vel_at_cop_[2]* ellipse_norm/ (ellipse[1])
            elif abs(ellipse[1]) <= 1e-10:
                beta_tau = 1e10
            else:
                beta_tau = 1

            if abs(ellipse[0]) > 1e-10 and v_norm_t != 0:
                beta_t = v_norm_t*ellipse_norm / (ellipse[0])
            elif abs(ellipse[0]) <= 1e-10:
                beta_t = 1e10
            else:
                beta_t = 1

            beta = np.array([beta_t, beta_t, beta_tau])

        self.lugre['z'] = vel_at_cop_ / (self.p['s0'] * (beta / g))

        scaling = np.array([1, 1, (9/8)])
        self.lugre['f'] = -(self.p['s0'] * self.lugre['z'] + scaling * self.p['s2'] * vel_at_cop_) * self.f_n
        self.lugre['f'][2] = self.lugre['f'][2]*self.gamma

    def update_lugre_ellipse(self, vel_at_cop, ellipse):
        vel_at_cop_ = np.array([vel_at_cop['x'], vel_at_cop['y'], vel_at_cop['tau']*self.gamma])
        v_norm = np.linalg.norm(vel_at_cop_)

        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * np.exp(- (v_norm/self.p['v_s'])**self.p['alpha'])

        v_norm_t = np.linalg.norm(vel_at_cop_[0:2])

        if vel_at_cop_[2] != 0:
            beta_tau = ellipse[1] * v_norm / vel_at_cop_[2]
        else:
            beta_tau = 1

        if v_norm_t != 0:
            beta_t = ellipse[0]*v_norm / v_norm_t
        else:
            beta_t = 1

        beta = np.array([beta_t, beta_t, beta_tau])

        scaling = np.array([1, 1, (9/8)])

        dz = beta*vel_at_cop_ - self.lugre['z'] * (self.p['s0'] * (v_norm / g))

        self.lugre['f'] = -(self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz + scaling * self.p['s2'] * vel_at_cop_) * self.f_n
        self.lugre['f'][2] = self.lugre['f'][2]*self.gamma
        self.lugre['z'] += dz * self.p['dt']

    def update_lugre_ellipse_stable(self, vel_at_cop, ellipse):
        vel_at_cop_ = np.array([vel_at_cop['x'], vel_at_cop['y'], vel_at_cop['tau']*self.gamma])
        v_norm = np.linalg.norm(vel_at_cop_)

        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * np.exp(- (v_norm/self.p['v_s'])**self.p['alpha'])

        v_norm_t = np.linalg.norm(vel_at_cop_[0:2])

        if vel_at_cop_[2] != 0:
            beta_tau = ellipse[1] * v_norm / vel_at_cop_[2]
        else:
            beta_tau = 1

        if v_norm_t != 0:
            beta_t = ellipse[0]*v_norm / v_norm_t
        else:
            beta_t = 1

        beta = np.array([beta_t, beta_t, beta_tau])

        scaling = np.array([1, 1, (9/8)])

        if v_norm == 0:
            v_norm1 = 1
        else:
            v_norm1 = v_norm

        z_ss_1 = vel_at_cop_ * g / (self.p['s0'] * v_norm1)
        z_ss = beta*z_ss_1
        delta_z = (z_ss - self.lugre['z']) / self.p['dt']

        dz = beta*vel_at_cop_ - self.lugre['z'] * (self.p['s0'] * (v_norm / g))

        dz = np.clip(abs(dz), np.zeros(dz.shape), abs(delta_z))*np.sign(dz)

        self.lugre['f'] = -(self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz + scaling * self.p['s2'] * vel_at_cop_) * self.f_n
        self.lugre['f'][2] = self.lugre['f'][2]*self.gamma
        self.lugre['z'] += dz * self.p['dt']

    def update_lugre_ellipse_stable_elasto_plastic(self, vel_at_cop, ellipse):
        vel_at_cop_ = np.array([vel_at_cop['x'], vel_at_cop['y'], vel_at_cop['tau']*self.gamma])
        v_norm = np.linalg.norm(vel_at_cop_)

        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * np.exp(- (v_norm/self.p['v_s'])**self.p['alpha'])

        v_norm_t = np.linalg.norm(vel_at_cop_[0:2])

        if vel_at_cop_[2] != 0:
            beta_tau = ellipse[1] * v_norm / vel_at_cop_[2]
        else:
            beta_tau = 1

        if v_norm_t != 0:
            beta_t = ellipse[0]*v_norm / v_norm_t
        else:
            beta_t = 1

        beta = np.array([beta_t, beta_t, beta_tau])

        scaling = np.array([1, 1, (9/8)])

        if v_norm == 0:
            v_norm1 = 1
        else:
            v_norm1 = v_norm

        z_ss_1 = vel_at_cop_ * g / (self.p['s0'] * v_norm1)
        z_ss = beta*z_ss_1

        z_max = np.linalg.norm(z_ss)
        z_norm = np.linalg.norm(self.lugre['z'])
        z_ba = 0.9*z_max
        if z_norm <= z_ba:
            alpha = 0
        elif z_norm <= z_max:
            alpha = 0.5 * np.sin((z_norm - (z_max - z_ba) / 2) / (z_max - z_ba)) + 0.5
        else:
            alpha = 1

        if v_norm != 0 and z_norm != 0:
            v_unit = vel_at_cop_ / v_norm
            z_unit = self.lugre['z'] / z_norm
            c = v_unit.dot(z_unit)
            eps = (c + 1) / 2
            alpha = eps * alpha

        delta_z = (z_ss - self.lugre['z']) / self.p['dt']

        dz = beta*vel_at_cop_ - alpha*self.lugre['z'] * (self.p['s0'] * (v_norm / g))
        dz2 = beta*vel_at_cop_ - self.lugre['z'] * (self.p['s0'] * (v_norm / g))
        dz = np.clip(abs(dz), np.zeros(dz.shape), abs(delta_z))*np.sign(dz2)

        self.lugre['f'] = -(self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz + scaling * self.p['s2'] * vel_at_cop_) * self.f_n
        self.lugre['f'][2] = self.lugre['f'][2]*self.gamma
        self.lugre['z'] += dz * self.p['dt']

    def move_force_to_center(self, force_at_cop):
        f_t = np.array([force_at_cop['x'], force_at_cop['y']])
        m = 0
        f_t_n = np.linalg.norm(f_t)
        if f_t_n != 0:
            d = np.linalg.norm(np.cross(-self.cop, f_t))/f_t_n
            if np.cross(f_t, -self.cop) > 0:  # left side of force
                m = d * f_t_n
            else:
                m = - d * f_t_n
        return {'x': force_at_cop['x'], 'y': force_at_cop['y'], 'tau': force_at_cop['tau'] + m, 'gamma': self.gamma}
