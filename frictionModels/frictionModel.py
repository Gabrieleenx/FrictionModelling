import numpy as np
import copy
from typing import Dict
from frictionModels.utils import elasto_plastic_alpha, CustomHashList3D, update_radius, update_viscus_scale, vel_to_cop

"""
properties = {'grid_shape': (21, 21),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.3,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e5,
              's1': 2e1,
              's2': 0.4,
              'dt': 1e-4,
              'stability': True,
              'elasto_plastic': True,
              'z_ba_ratio': 0.9,
              'steady_state': False}
"""

class FrictionBase(object):
    def __init__(self, properties: Dict[str, any]):
        self.p = properties
        self.cop = np.zeros(2)
        self.normal_force_grid = np.zeros((self.p['grid_shape'][0], self.p['grid_shape'][1]))
        self.fn = 0
        self.x_pos_vec = self.get_pos_vector(0)
        self.y_pos_vec = self.get_pos_vector(0)
        self.pos_matrix, self.pos_matrix_2d = self.get_pos_matrix()
        self.force_at_cop = None

    def get_pos_vector(self, i):
        return (np.arange(self.p['grid_shape'][i]) + 0.5 - self.p['grid_shape'][i] / 2) * self.p['grid_size']

    def get_pos_matrix(self):
        pos_matrix = np.zeros((self.p['grid_shape'][0], self.p['grid_shape'][1], 3))

        for i_x in range(self.p['grid_shape'][0]):
            for i_y in range(self.p['grid_shape'][1]):
                pos_matrix[i_x, i_y, 0] = self.x_pos_vec[i_x]
                pos_matrix[i_x, i_y, 1] = self.y_pos_vec[i_y]

        pos_matrix_2d = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))
        pos_matrix_2d[0, :, :] = pos_matrix[:, :, 0]
        pos_matrix_2d[1, :, :] = pos_matrix[:, :, 1]
        return pos_matrix, pos_matrix_2d

    def update_cop_and_force_grid(self, p_x_y):
        area = self.p['grid_size'] ** 2
        self.normal_force_grid = p_x_y(self.pos_matrix_2d)
        self.normal_force_grid = self.normal_force_grid * area
        self.fn = np.sum(self.normal_force_grid)
        self.cop[0] = np.sum(self.x_pos_vec.dot(self.normal_force_grid)) / self.fn
        self.cop[1] = np.sum(self.y_pos_vec.dot(self.normal_force_grid.T)) / self.fn

    def move_force_to_center(self, force_at_cop):
        f_t = np.array([force_at_cop['x'], force_at_cop['y']])
        m = np.cross(self.cop, f_t)
        return {'x': force_at_cop['x'], 'y': force_at_cop['y'], 'tau': force_at_cop['tau'] + m}

    def move_force_to_cop(self, force_at_center):
        f_t = np.array([force_at_center['x'], force_at_center['y']])
        m = np.cross(self.cop, f_t)
        return {'x': force_at_center['x'], 'y': force_at_center['y'], 'tau': force_at_center['tau'] - m}

    def update_properties(self,
                          mu_c: float = None,
                          mu_s: float = None,
                          v_s: float = None,
                          alpha: float = None,
                          s0: float = None,
                          s1: float = None,
                          s2: float = None,
                          dt: float = None,
                          stability: bool = None,
                          elasto_plastic: bool = None,
                          z_ba_ratio: float = None,
                          steady_state: bool = None):
        """
        Function to update properties
        :param mu_c:
        :param mu_s:
        :param v_s:
        :param alpha:
        :param s0:
        :param s1:
        :param s2:
        :param dt:
        :param stability:
        :param elasto_plastic:
        :param z_ba_ratio:
        :param steady_state:
        :return:
        """
        self.p['mu_c'] = mu_c if (mu_c is not None) else self.p['mu_c']
        self.p['mu_s'] = mu_s if (mu_s is not None) else self.p['mu_s']
        self.p['v_s'] = v_s if (v_s is not None) else self.p['v_s']
        self.p['alpha'] = alpha if (alpha is not None) else self.p['alpha']
        self.p['s0'] = s0 if (s0 is not None) else self.p['s0']
        self.p['s1'] = s1 if (s1 is not None) else self.p['s1']
        self.p['s2'] = s2 if (s2 is not None) else self.p['s2']
        self.p['dt'] = dt if (dt is not None) else self.p['dt']
        self.p['stability'] = stability if (stability is not None) else self.p['stability']
        self.p['elasto_plastic'] = elasto_plastic if (elasto_plastic is not None) else self.p['elasto_plastic']
        self.p['z_ba_ratio'] = z_ba_ratio if (z_ba_ratio is not None) else self.p['z_ba_ratio']
        self.p['steady_state'] = steady_state if (steady_state is not None) else self.p['steady_state']


class FullFrictionModel(FrictionBase):
    def __init__(self, properties: Dict[str, any]):
        super().__init__(properties)
        self.velocity_grid = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))
        self.lugre = self.initialize_lugre()

    def step(self, vel_vec: Dict[str, float]) -> Dict[str, float]:
        """
        This function does one time step of the friction model
        :param vel_vec:
        :return:
        """
        self.update_velocity_grid(vel_vec)
        self.update_lugre()
        force_at_center = self.approximate_integral()
        self.force_at_cop = self.move_force_to_cop(force_at_center)

        #force = self.move_force_to_center(self.force_at_cop)

        return force_at_center

    def update_p_x_y(self, p_x_y):
        self.update_cop_and_force_grid(p_x_y)

    def update_velocity_grid(self, vel_vec):
        u = np.array([0, 0, 1])
        w = vel_vec['tau'] * u
        v_tau = np.cross(w, self.pos_matrix)
        self.velocity_grid[0, :, :] = v_tau[:, :, 0] + vel_vec['x']
        self.velocity_grid[1, :, :] = v_tau[:, :, 1] + vel_vec['y']

    def initialize_lugre(self):
        z = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))  # bristles
        f = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))  # tangential force at each grid cell
        return {'z': z, 'f': f}

    def update_lugre(self):
        v_norm = np.linalg.norm(self.velocity_grid, axis=0)
        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * np.exp(- (v_norm / self.p['v_s']) ** self.p['alpha'])
        z_ss = self.steady_state_z(v_norm, g)

        if self.p['steady_state']:
            self.lugre['f'] = (self.p['s0'] * z_ss + self.p[
                's2'] * self.velocity_grid) * self.normal_force_grid
            return

        if self.p['elasto_plastic']:
            alpha = self.elasto_plastic(z_ss)
            dz = self.velocity_grid - alpha * self.lugre['z'] * (self.p['s0'] * (v_norm / g))
        else:
            dz = self.velocity_grid - self.lugre['z'] * (self.p['s0'] * (v_norm / g))

        if self.p['stability']:
            delta_z = (z_ss - self.lugre['z']) / self.p['dt']
            dz = np.min([abs(dz), abs(delta_z)], axis=0)*np.sign(dz)

        self.lugre['f'] = (self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz +
                           self.p['s2'] * self.velocity_grid) * self.normal_force_grid

        self.lugre['z'] += dz * self.p['dt']

    def steady_state_z(self, v_norm, g):
        v_norm1 = v_norm.copy()
        v_norm1[v_norm1 == 0] = 1
        lugre_ss = self.velocity_grid*g / (self.p['s0'] * v_norm1)
        return lugre_ss

    def elasto_plastic(self, z_ss):
        alpha = np.zeros(self.p['grid_shape'])

        for i_x, x_ in enumerate(self.x_pos_vec):
            for i_y, y_ in enumerate(self.y_pos_vec):
                alpha[i_x, i_y] = elasto_plastic_alpha(self.lugre['z'][:, i_x, i_y],
                                                       z_ss[:, i_x, i_y],
                                                       self.p['z_ba_ratio'],
                                                       self.velocity_grid[:, i_x, i_y])

        return alpha

    def approximate_integral(self):
        fx = - np.sum(self.lugre['f'][0, :, :])
        fy = - np.sum(self.lugre['f'][1, :, :])
        tau = np.cross(self.pos_matrix_2d, self.lugre['f'], axis=0)
        return {'x': fx, 'y': fy, 'tau': -np.sum(tau)}


class ReducedFrictionModel(FrictionBase):
    def __init__(self, properties: Dict[str, any], nr_ls_segments: int = 20, ls_active: bool = True):
        super().__init__(properties)
        self.gamma = 0.00764477848712988
        self.limit_surface = CustomHashList3D(nr_ls_segments)
        self.viscous_scale = np.ones(3)
        self.p_x_y = None
        self.full_model = self.initialize_full_model()
        self.lugre = self.initialize_lugre()
        self.ls_active = ls_active

    def step(self, vel_vec: Dict[str, float]) -> Dict[str, float]:
        """
        This function does one time step of the friction model
        :param vel_vec:
        :return:
        """

        vel_cop = vel_to_cop(self.cop, vel_vec)

        self.update_lugre(vel_cop)

        force_at_cop = {'x': self.lugre['f'][0], 'y': self.lugre['f'][1], 'tau': self.lugre['f'][2]}

        force = self.move_force_to_center(force_at_cop)

        return force


    def initialize_full_model(self):
        properties = copy.deepcopy(self.p)
        properties['mu_c'] = 1
        properties['mu_s'] = 1
        properties['s2'] = 0
        properties['steady_state'] = True
        return FullFrictionModel(properties)

    def initialize_lugre(self):
        z = np.zeros(3)  # bristles
        f = np.zeros(3)  # tangential force at each grid cell
        return {'z': z, 'f': f}

    def update_p_x_y(self, p_x_y):
        self.p_x_y = p_x_y
        self.update_cop_and_force_grid(p_x_y)

    def update_pre_compute(self):
        # limit surface
        self.full_model.update_p_x_y(self.p_x_y)
        #self.limit_surface.initialize(self.full_model, self.cop)
        self.limit_surface.initialize(self.full_model)
        # gamma radius
        self.gamma = update_radius(self.full_model)
        # viscus scale
        self.viscous_scale = update_viscus_scale(self.full_model, self.gamma, self.cop)

    def update_lugre(self, vel_cop):
        vel_cop_tau = np.array([vel_cop['x'], vel_cop['y'], vel_cop['tau']*self.gamma])
        v_norm = np.linalg.norm(vel_cop_tau)

        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * np.exp(- (v_norm / self.p['v_s']) ** self.p['alpha'])
        if self.ls_active:
            beta = self.calc_beta(vel_cop, vel_cop_tau, v_norm)
        else:
            beta = np.ones(3)

        if v_norm != 0:
            z_ss = (beta*vel_cop_tau*g) / (self.p['s0'] * v_norm)
        else:
            z_ss = np.zeros(3)

        if self.p['steady_state']:
            self.lugre['f'] = -(self.p['s0'] * z_ss + self.viscous_scale * self.p['s2'] * vel_cop_tau) * self.fn
            self.lugre['f'][2] = self.lugre['f'][2] * self.gamma
            return


        if self.p['elasto_plastic']:
            alpha = elasto_plastic_alpha(self.lugre['z'],
                                         z_ss,
                                         self.p['z_ba_ratio'],
                                         vel_cop_tau)
        else:
            alpha = 1
        dz = beta * vel_cop_tau - alpha * self.lugre['z'] * (self.p['s0'] * (v_norm / g))

        if self.p['stability']:
            delta_z = (z_ss - self.lugre['z']) / self.p['dt']

            dz = np.min([abs(dz), abs(delta_z)], axis=0) * np.sign(dz)

        self.lugre['f'] = -(self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz +
                            self.viscous_scale * self.p['s2'] * vel_cop_tau) * self.fn
        self.lugre['f'][2] = self.lugre['f'][2]*self.gamma

        self.lugre['z'] += dz * self.p['dt']

    """
    def calc_beta(self, vel_cop, vel_cop_tau, v_norm):
        ls_surf = self.limit_surface.get_interpolation(np.linalg.norm([vel_cop['x'], vel_cop['y']]),
                                                       abs(vel_cop['tau']))
        v_norm_t = np.linalg.norm(vel_cop_tau[0:2])

        if vel_cop_tau[2] != 0:
            beta_tau = ls_surf[1] * v_norm / abs(vel_cop_tau[2])
        else:
            beta_tau = 1

        if v_norm_t != 0:
            beta_t = ls_surf[0]*v_norm / v_norm_t
        else:
            beta_t = 1

        return np.array([beta_t, beta_t, beta_tau])
    """

    def calc_beta(self, vel_cop, vel_cop_tau, v_norm):
        ls = self.limit_surface.get_bilinear_interpolation(vel_cop, self.gamma)
        beta = np.zeros(3)
        for i in range(3):
            if vel_cop_tau[i] != 0:
                beta[i] = abs(ls[i]) * v_norm / abs(vel_cop_tau[i])
            else:
                beta[i] = 1
        return beta


    def steady_state(self, vel_cop, force_at_cop):
        force_vec = {}
        ratio = 0
        return force_vec, ratio

