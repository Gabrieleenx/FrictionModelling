import numpy
import numpy as np
from typing import Dict, Callable

"""
properties = {'grid_shape': (0, 0),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.3,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e4,
              's1': 2e1,
              's2': 0.4,
              'dt': 1e-4}
"""


class PlanarFriction(object):
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
        z = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))  # bristles
        f = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))  # tangential force at each grid cell
        self.lugre = {'z': z, 'f': f}

    def step(self, vel_vec: Dict[str, float], p_x_y: Callable[[float, float], float]) -> Dict[str, float]:
        """
        k
        :param vel_vec: velocity vector in the center of the censor
        :param p_x_y: function that takes x, y and returns p
        :return: force in x, y and moment expressed in the center of the sensor.
        """

        self.update_cop_and_force_grid(p_x_y)

        self.update_velocity_grid(vel_vec)

        self.update_lugre()

        force_at_cop = self.approximate_integral()

        force = self.move_force_to_center(force_at_cop)

        return force

    def step_stability(self, vel_vec: Dict[str, float], p_x_y: Callable[[float, float], float]) -> Dict[str, float]:
        """
        k
        :param vel_vec: velocity vector in the center of the censor
        :param p_x_y: function that takes x, y and returns p
        :return: force in x, y and moment expressed in the center of the sensor.
        """

        self.update_cop_and_force_grid(p_x_y)

        self.update_velocity_grid(vel_vec)

        self.update_lugre_stability()

        force_at_cop = self.approximate_integral()

        force = self.move_force_to_center(force_at_cop)

        return force

    def step_stability_elasto_plastic(self, vel_vec: Dict[str, float], p_x_y: Callable[[float, float], float]) -> Dict[str, float]:
        """
        k
        :param vel_vec: velocity vector in the center of the censor
        :param p_x_y: function that takes x, y and returns p
        :return: force in x, y and moment expressed in the center of the sensor.
        """

        self.update_cop_and_force_grid(p_x_y)

        self.update_velocity_grid(vel_vec)

        self.update_lugre_stability_elasto_plastic()

        force_at_cop = self.approximate_integral()

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

        self.update_velocity_grid(vel_vec)

        self.update_lugre_ss()

        force_at_cop = self.approximate_integral()

        force = self.move_force_to_center(force_at_cop)

        return force

    def update_cop_and_force_grid(self, p_x_y):
        area = self.p['grid_size']**2

        """
        for i_x in range(self.p['grid_shape'][0]):
            for i_y in range(self.p['grid_shape'][1]):
                self.normal_force_grid[i_x, i_y] = p_x_y(self.x_pos_vec[i_x], self.y_pos_vec[i_y])
        """

        self.normal_force_grid = p_x_y(self.pos_matrix_2d)

        #self.normal_force_grid = np.ones((self.p['grid_shape'][0], self.p['grid_shape'][1])) * p_x_y(0, 0)

        self.normal_force_grid = self.normal_force_grid * area
        f_n = np.sum(self.normal_force_grid)

        self.cop[0] = np.sum(self.x_pos_vec.dot(self.normal_force_grid))/f_n
        self.cop[1] = np.sum(self.y_pos_vec.dot(self.normal_force_grid.T))/f_n

    def update_velocity_grid(self, vel_vec):
        u = np.array([0, 0, 1])
        w = vel_vec['tau'] * u
        v_tau = np.cross(w, self.pos_matrix)
        self.velocity_grid[0, :, :] = v_tau[:, :, 0] + vel_vec['x']
        self.velocity_grid[1, :, :] = v_tau[:, :, 1] + vel_vec['y']

    def update_lugre(self):

        v_norm = np.linalg.norm(self.velocity_grid, axis=0)
        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * \
            np.exp(- (v_norm/self.p['v_s'])**self.p['alpha'])

        dz = self.velocity_grid - self.lugre['z'] * (self.p['s0'] * (v_norm / g))

        self.lugre['f'] = (self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz + self.p['s2'] * self.velocity_grid) * self.normal_force_grid

        self.lugre['z'] += dz * self.p['dt']

    def update_lugre_stability(self):

        v_norm = np.linalg.norm(self.velocity_grid, axis=0)
        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * \
            np.exp(- (v_norm / self.p['v_s']) ** self.p['alpha'])

        v_norm1 = v_norm.copy()
        v_norm1[v_norm1 == 0] = 1
        luGre_ss = self.velocity_grid / (self.p['s0'] * (v_norm1 / g))
        delta_z = (luGre_ss - self.lugre['z']) / self.p['dt']

        dz = self.velocity_grid - self.lugre['z'] * (self.p['s0'] * (v_norm / g))

        dz = np.clip(abs(dz), np.zeros((2, 20, 20)), abs(delta_z)) * np.sign(dz)
        self.lugre['f'] = (self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz + self.p[
            's2'] * self.velocity_grid) * self.normal_force_grid

        self.lugre['z'] += dz * self.p['dt']

    def update_lugre_stability_elasto_plastic(self):

        v_norm = np.linalg.norm(self.velocity_grid, axis=0)
        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * \
            np.exp(- (v_norm / self.p['v_s']) ** self.p['alpha'])

        v_norm1 = v_norm.copy()
        v_norm1[v_norm1 == 0] = 1
        luGre_ss = self.velocity_grid*g / (self.p['s0'] * v_norm1)
        delta_z = (luGre_ss - self.lugre['z']) / self.p['dt']

        alpha = np.zeros(self.p['grid_shape'])

        for i_x, x_ in enumerate(self.x_pos_vec):
            for i_y, y_ in enumerate(self.y_pos_vec):
                z_norm = np.linalg.norm(self.lugre['z'][:, i_x, i_y])
                z_max = g[i_x, i_y]/self.p['s0']
                z_ba = 0.9*z_max
                if z_norm <= z_ba:
                    alpha[i_x, i_y] = 0
                elif z_norm <= z_max:
                    alpha[i_x, i_y] = 0.5 * np.sin((z_norm - (z_max - z_ba)/2)/(z_max-z_ba)) + 0.5
                else:
                    alpha[i_x, i_y] = 1

                if v_norm[i_x, i_y] != 0 and z_norm != 0:
                    v_unit = self.velocity_grid[:, i_x, i_y] / v_norm[i_x, i_y]
                    z_unit = self.lugre['z'][:, i_x, i_y] / z_norm
                    c = v_unit.dot(z_unit)
                    eps = (c+1)/2
                    alpha[i_x, i_y] = eps*alpha[i_x, i_y]



        dz = self.velocity_grid - alpha*self.lugre['z'] * (self.p['s0'] * (v_norm / g))

        dz_2 = self.velocity_grid - self.lugre['z'] * (self.p['s0'] * (v_norm / g))


        dz = np.clip(abs(dz), np.zeros((2, 20, 20)), abs(delta_z)) * np.sign(dz_2)
        self.lugre['f'] = (self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz + self.p[
            's2'] * self.velocity_grid) * self.normal_force_grid

        self.lugre['z'] += dz * self.p['dt']

    def update_lugre_ss(self):
        v_norm = np.linalg.norm(self.velocity_grid, axis=0)
        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * \
            np.exp(- (v_norm / self.p['v_s']) ** self.p['alpha'])
        v_norm[v_norm==0] = 1
        self.lugre['z'] = self.velocity_grid / (self.p['s0'] * (v_norm / g))

        self.lugre['f'] = (self.p['s0'] * self.lugre['z'] + self.p['s2'] * self.velocity_grid) * self.normal_force_grid

    def approximate_integral(self):
        fx = - np.sum(self.lugre['f'][0, :, :])
        fy = - np.sum(self.lugre['f'][1, :, :])
        tau = np.cross(self.pos_matrix_2d, self.lugre['f'], axis=0)
        return {'x': fx, 'y': fy, 'tau': -np.sum(tau)}

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
        return {'x': force_at_cop['x'], 'y': force_at_cop['y'], 'tau': force_at_cop['tau'] + m}
