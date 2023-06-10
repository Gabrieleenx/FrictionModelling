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
        self.y_pos_vec = self.get_pos_vector(1)
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
                          grid_size: float = None,
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
        self.p['grid_size'] = grid_size if (grid_size is not None) else self.p['grid_size']
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
        self.p_x_y = None

    def step_single_point(self, vel_vec: Dict[str, float]) -> Dict[str, float]:
        """
        This function does one time step of the friction model
        :param vel_vec:
        :return:
        """
        self.update_velocity_grid(vel_vec)
        self.update_lugre()
        force_at_center = self.approximate_integral()
        self.force_at_cop = self.move_force_to_cop(force_at_center)

        return force_at_center

    def set_fn(self, fn):
        self.p_x_y.set_fn(fn)
        self.fn = fn

    def step(self, vel_vec: Dict[str, float]) -> Dict[str, float]:
        self.normal_force_grid, self.cop, self.fn = self.p_x_y.get(self.p['grid_size'])

        if not self.p['steady_state']:
            return self.step_single_point(vel_vec)
        else:
            vx = vel_vec['x']
            vy = vel_vec['y']
            vtau = vel_vec['tau']
            if vtau == 0:
                self.update_velocity_grid(vel_vec)
                self.update_lugre()
                force_at_center = self.approximate_integral()
            else:

                nx = self.p['grid_shape'][0]
                ny = self.p['grid_shape'][1]
                x0 = nx * self.p['grid_size']/2
                y0 = ny * self.p['grid_size']/2

                cor = np.array([-vy/vtau, vx/vtau])


                ix_ = nx*(cor[0] + x0)/(2*abs(x0))
                ix = int(ix_)
                dx = ix_ - ix

                iy_ = ny*(cor[1] + y0)/(2*abs(y0))
                iy = int(iy_)
                dy = iy_ - iy


                if ix >= 0 and ix < nx and iy >= 0 and iy < ny:
                    ox = [0, 1, 0, 1]
                    oy = [0, 0, 1, 1]

                    f = [0,0,0,0]

                    for i in range(4):
                        corx = ((ix + ox[i])/nx) * (2*abs(x0)) - x0
                        cory = ((iy + oy[i]) / ny) * (2 * abs(y0)) - y0
                        cor_ = np.array([corx, cory])
                        vel_new = vel_to_cop(-cor_, {'x':0, 'y':0, 'tau':vtau})

                        self.update_velocity_grid(vel_new)
                        self.update_lugre()
                        f[i] = self.approximate_integral()

                    s1 = (1 - dx) * (1 - dy)
                    s2 = dx * (1 - dy)
                    s3 = dy * (1 - dx)
                    s4 = dx * dy
                    fx = s1*f[0]['x'] + s2*f[1]['x'] + s3*f[2]['x'] + s4*f[3]['x']
                    fy = s1*f[0]['y'] + s2*f[1]['y'] + s3*f[2]['y'] + s4*f[3]['y']
                    ftau = s1*f[0]['tau'] + s2*f[1]['tau'] + s3*f[2]['tau'] + s4*f[3]['tau']

                    force_at_center = {'x':fx, 'y':fy, 'tau':ftau}

                else:
                    self.update_velocity_grid(vel_vec)
                    self.update_lugre()
                    force_at_center = self.approximate_integral()

            self.force_at_cop = self.move_force_to_cop(force_at_center)

            return force_at_center
    def update_p_x_y(self, p_x_y):
        #self.update_cop_and_force_grid(p_x_y)
        self.p_x_y = p_x_y
        self.p['grid_size'] = self.p_x_y.size
        self.x_pos_vec = self.get_pos_vector(0)
        self.y_pos_vec = self.get_pos_vector(1)
        self.pos_matrix, self.pos_matrix_2d = self.get_pos_matrix()
        self.normal_force_grid, self.cop, self.fn = self.p_x_y.get(self.p['grid_size'])


    def update_velocity_grid(self, vel_vec):
        u = np.array([0, 0, 1])
        w = vel_vec['tau'] * u
        v_tau = np.cross(w, self.pos_matrix)
        self.velocity_grid[0, :, :] = v_tau[:, :, 0] + vel_vec['x']
        self.velocity_grid[1, :, :] = v_tau[:, :, 1] + vel_vec['y']

    def initialize_lugre(self):
        z = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))  # bristles
        f = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))  # tangential force at each grid cell
        dz = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))  # derivative of bristles
        return {'z': z, 'f': f, 'dz': dz}

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

        self.lugre['z'] += dz * self.p['dt']

        self.lugre['f'] = (self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz +
                           self.p['s2'] * self.velocity_grid) * self.normal_force_grid
        self.lugre['dz'] = dz

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

    def ode_step(self, t, y, vel):
        shape_ = self.lugre['z'].shape

        self.lugre['z'] = np.reshape(y, shape_)
        vel_vec = {'x': vel[0], 'y': vel[1], 'tau': vel[2]}
        f = self.step_single_point(vel_vec)
        dy = self.lugre['dz']
        return np.reshape(dy, np.prod(shape_)).tolist(), f

    def ode_init(self):
        shape_ = self.lugre['z'].shape
        return np.reshape(self.lugre['z'], np.prod(shape_)).tolist()


class ReducedFrictionModel(FrictionBase):
    def __init__(self, properties: Dict[str, any], nr_ls_segments: int = 20, ls_active: bool = True):
        super().__init__(properties)
        self.ra = 0.00764477848712988
        self.limit_surface = CustomHashList3D(nr_ls_segments)
        self.viscous_scale = np.ones(3)
        self.p_x_y = None
        self.full_model = self.initialize_full_model()
        self.lugre = self.initialize_lugre()
        self.ls_active = ls_active

    def set_fn(self, fn):
        self.p_x_y.set_fn(fn)

    def step(self, vel_vec: Dict[str, float]) -> Dict[str, float]:
        """
        This function does one time step of the friction model
        :param vel_vec:
        :return:
        """
        p, self.cop, self.fn = self.p_x_y.get(self.p['grid_size'])

        vel_cop = vel_to_cop(self.cop, vel_vec)

        self.update_lugre(vel_cop)

        self.force_at_cop = {'x': self.lugre['f'][0 ,0], 'y': self.lugre['f'][1,0], 'tau': self.lugre['f'][2,0]}

        force = self.move_force_to_center(self.force_at_cop)

        return force


    def initialize_full_model(self):
        properties = copy.deepcopy(self.p)
        properties['mu_c'] = 1
        properties['mu_s'] = 1
        properties['s2'] = 0
        properties['steady_state'] = True
        return FullFrictionModel(properties)

    def initialize_lugre(self):
        z = np.zeros((3, 1))  # bristles
        f = np.zeros((3, 1))  # tangential force at each grid cell
        return {'z': z, 'f': f}

    def update_p_x_y(self, p_x_y_object):
        self.p_x_y = p_x_y_object
        self.full_model.update_p_x_y(self.p_x_y)
        self.p['grid_size'] = self.p_x_y.size

        # gamma radius
        self.ra = update_radius(self.full_model)
        # viscus scale
        self.viscous_scale = update_viscus_scale(self.full_model, self.ra, self.cop)
        #self.update_cop_and_force_grid(p_x_y)

    def update_pre_compute(self):
        # limit surface
        #self.full_model.update_p_x_y(self.p_x_y)
        self.limit_surface.initialize(self.full_model)
        # gamma radius
        #self.ra = update_radius(self.full_model)
        # viscus scale
        #self.viscous_scale = update_viscus_scale(self.full_model, self.ra, self.cop)

    def update_lugre(self, vel_cop):
        vel_cop_list = np.array([[vel_cop['x']], [vel_cop['y']], [vel_cop['tau']]])

        if self.ls_active:
            S_A, inv_A = self.calc_beta(vel_cop)
        else:
            inv_A = np.eye(3)
            inv_A[2, 2] = self.ra**2
            S_A = np.eye(3)
        v_norm = np.sqrt(vel_cop_list.T.dot(inv_A).dot(vel_cop_list))
        inv_A_v = inv_A.dot(vel_cop_list)

        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * np.exp(- (v_norm / self.p['v_s']) ** self.p['alpha'])

        if v_norm != 0:
            z_ss = (S_A.dot(inv_A_v)*g) / (self.p['s0'] * v_norm)

        else:
            z_ss = np.zeros((3, 1))
        if self.p['steady_state']:
            self.lugre['f'] = -(self.p['s0'] * z_ss + self.viscous_scale * self.p['s2'] * inv_A_v) * self.fn

            return


        if self.p['elasto_plastic']:
            R_a = np.diag([1, 1, 1/self.ra])
            vel_cop_tau = R_a.dot(S_A).dot(inv_A_v).flatten()
            alpha = elasto_plastic_alpha(R_a.dot(self.lugre['z']).flatten(),
                                         R_a.dot(z_ss).flatten(),
                                         self.p['z_ba_ratio'],
                                         vel_cop_tau)

        else:
            alpha = 1

        dz = S_A.dot(inv_A_v) - alpha * self.lugre['z'] * (self.p['s0'] * (v_norm / g))
        if self.p['stability']:
            z = self.lugre['z']
            delta_z = (z_ss - z) / self.p['dt']

            dz = np.min([abs(dz), abs(delta_z)], axis=0) * np.sign(dz)
        self.lugre['z'] += dz * self.p['dt']

        f = -(self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz +
                            self.viscous_scale * self.p['s2'] * inv_A_v) * self.fn
        self.lugre['f'] = f
        self.lugre['dz'] = dz

    def calc_beta(self, vel_cop):
        vel_cop_list = np.array([[vel_cop['x']], [vel_cop['y']], [vel_cop['tau']]])
        ls = self.limit_surface.get_bilinear_interpolation(vel_cop, self.ra)

        inv_A = np.eye(3)
        inv_A[2, 2] = self.ra ** 2
        new_vel = self.limit_surface.calc_new_vel(vel_cop, self.ra)

        if vel_cop['tau'] != 0:
            sx = (new_vel['x'] - vel_cop['x'])/vel_cop['tau']
            sy = (new_vel['y'] - vel_cop['y'])/vel_cop['tau']
        else:
            sx = 1
            sy = 1
        inv_A[0, 2] = sx
        inv_A[1, 2] = sy

        v_a = abs(inv_A.dot(vel_cop_list))
        v_a[v_a == 0] = 1
        V_A = np.diag(v_a.flatten())
        LS = np.diag(ls)
        LS[2,2] = self.ra * LS[2,2]
        S_A = abs(LS).dot(np.linalg.pinv(V_A)) * np.sqrt(vel_cop_list.T.dot(inv_A).dot(vel_cop_list))
        return S_A, inv_A


    def steady_state(self, vel_cop, force_at_cop):
        force_vec = {}
        ratio = 0
        return force_vec, ratio


    def ode_step(self, t, y, vel):
        shape_ = self.lugre['z'].shape

        self.lugre['z'] = np.reshape(y, shape_)
        vel_vec = {'x': vel[0], 'y': vel[1], 'tau': vel[2]}
        f = self.step(vel_vec)
        dy = self.lugre['dz']
        return np.reshape(dy, np.prod(shape_)).tolist(), f

    def ode_init(self):
        shape_ = self.lugre['z'].shape
        return np.reshape(self.lugre['z'], np.prod(shape_)).tolist()


class LuGre1D(object):
    def __init__(self, properties: Dict[str, any], fn: float):
        self.p = properties
        self.fn = fn

    def set_fn(self, fn):
        self.fn = fn

    def ode_step(self, t, y, vel):

        z = y[0]
        dx = vel[0]

        # Parameters
        my_d = self.p["mu_c"]
        my_s = self.p["mu_s"]
        dx_s = self.p["v_s"]
        sigma_0 = self.p["s0"]
        sigma_1 = self.p["s1"]
        sigma_2 = self.p["s2"]
        z_ba_r = self.p["z_ba_ratio"]
        alpha = self.p["alpha"]

        if abs(dx) == 0:
            z_ss = my_s / sigma_0
        else:
            f_ss = ((my_s - my_d) * np.exp(-(dx / dx_s) ** alpha) + my_d) * np.sign(dx)
            z_ss = f_ss / sigma_0

        if self.p['elasto_plastic']:
            if abs(z) <= z_ba_r*abs(z_ss):
                alpha = 0
            elif abs(z) >= abs(z_ss):
                alpha = 1
            else:
                alpha = 0.5 * np.sin(np.pi * (abs(z) - (z_ss+z_ba_r*z_ss) / 2) / (z_ss - z_ba_r*z_ss)) + 0.5
        else:
            alpha = 1

        dz = (1 - alpha * z / z_ss) * dx

        mu = sigma_0 * z + sigma_1 * dz + sigma_2 * dx

        fx = - mu * self.fn
        f = {'x': fx, 'y': 0, 'tau': 0}

        return [dz], f

    def ode_init(self):
        return [0]

