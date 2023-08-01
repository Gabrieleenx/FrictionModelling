import numpy as np
import copy
from typing import Dict
from frictionModels.utils import elasto_plastic_beta, CustomHashList3D, update_radius, update_viscus_scale, vel_to_cop

"""
properties = {'grid_shape': (21, 21),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1, # Coulomb friction coefficient 
              'mu_s': 1.2, # static friction coefficient 
              'v_s': 1e-3, # Stribeck velocity 
              'alpha': 2, # Coefficient (called gamma in paper)
              's0': 1e6, # LuGre stiffness  
              's1': 8e1, # LuGre dampening 
              's2': 0.2, # vicious friction coefficient 
              'dt': 1e-4, # Time step
              'stability': True, # bool for limiting the bristle deflection rate
              'elasto_plastic': True, # bool for activating the elasto plastic extension 
              'z_ba_ratio': 0.9, # ratio for when the elasto-plastic model transitions to plastic deformation  
              'steady_state': False # bool for if the steady state friction should be returned}
"""


class FrictionBase(object):
    """
    Base class that will be inherited by the classes below.
    """
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


class DistributedFrictionModel(FrictionBase):
    """
    Class for simulating the distributed friction model.
    """
    def __init__(self, properties: Dict[str, any]):
        super().__init__(properties)
        self.velocity_grid = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))
        self.lugre = self.initialize_lugre()
        self.p_x_y = None

    def step_single_point(self, vel_vec: Dict[str, float]) -> Dict[str, float]:
        """
        This function returns the forces after one time step.
        :param vel_vec: dict with the velocities for the center of the contact patch
        :return: dict with forces at the center of the contact patch
        """
        self.update_velocity_grid(vel_vec)
        self.update_lugre()
        force_at_center = self.approximate_integral()
        self.force_at_cop = self.move_force_to_cop(force_at_center)

        return force_at_center

    def set_fn(self, fn):
        """
        Updates the normal force of the pressure distribution.
        :param fn: double
        """
        self.p_x_y.set_fn(fn)
        self.fn = fn

    def step(self, vel_vec: Dict[str, float]) -> Dict[str, float]:
        """
        This function returns the forces after one time step. For the steady state case then it returns the forces with
        bi-linear interpolation.
        :param vel_vec: dict with the velocities for the center of the contact patch
        :return: dict with forces at the center of the contact patch
        """
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
                x0 = nx * self.p['grid_size'] / 2
                y0 = ny * self.p['grid_size'] / 2

                cor = np.array([-vy / vtau, vx / vtau])

                ix_ = nx * (cor[0] + x0) / (2 * abs(x0))
                ix = int(ix_)
                dx = ix_ - ix

                iy_ = ny * (cor[1] + y0) / (2 * abs(y0))
                iy = int(iy_)
                dy = iy_ - iy

                if ix >= 0 and ix < nx and iy >= 0 and iy < ny:
                    ox = [0, 1, 0, 1]
                    oy = [0, 0, 1, 1]

                    f = [0, 0, 0, 0]

                    for i in range(4):
                        corx = ((ix + ox[i]) / nx) * (2 * abs(x0)) - x0
                        cory = ((iy + oy[i]) / ny) * (2 * abs(y0)) - y0
                        cor_ = np.array([corx, cory])
                        vel_new = vel_to_cop(-cor_, {'x': 0, 'y': 0, 'tau': vtau})

                        self.update_velocity_grid(vel_new)
                        self.update_lugre()
                        f[i] = self.approximate_integral()

                    s1 = (1 - dx) * (1 - dy)
                    s2 = dx * (1 - dy)
                    s3 = dy * (1 - dx)
                    s4 = dx * dy
                    fx = s1 * f[0]['x'] + s2 * f[1]['x'] + s3 * f[2]['x'] + s4 * f[3]['x']
                    fy = s1 * f[0]['y'] + s2 * f[1]['y'] + s3 * f[2]['y'] + s4 * f[3]['y']
                    ftau = s1 * f[0]['tau'] + s2 * f[1]['tau'] + s3 * f[2]['tau'] + s4 * f[3]['tau']

                    force_at_center = {'x': fx, 'y': fy, 'tau': ftau}

                else:
                    self.update_velocity_grid(vel_vec)
                    self.update_lugre()
                    force_at_center = self.approximate_integral()

            self.force_at_cop = self.move_force_to_cop(force_at_center)

            return force_at_center

    def update_p_x_y(self, p_x_y):
        """
        Updates the pressure distribution
        :param p_x_y: Object of type PObject for surfaces.py
        """
        self.p_x_y = p_x_y
        self.p['grid_size'] = self.p_x_y.size
        self.x_pos_vec = self.get_pos_vector(0)
        self.y_pos_vec = self.get_pos_vector(1)
        self.pos_matrix, self.pos_matrix_2d = self.get_pos_matrix()
        self.normal_force_grid, self.cop, self.fn = self.p_x_y.get(self.p['grid_size'])

    def update_velocity_grid(self, vel_vec):
        """
        Internal funtion to calculate all velocity vector as all points in the discretized surface
        :param vel_vec:
        """
        u = np.array([0, 0, 1])
        w = vel_vec['tau'] * u
        v_tau = np.cross(w, self.pos_matrix)
        self.velocity_grid[0, :, :] = v_tau[:, :, 0] + vel_vec['x']
        self.velocity_grid[1, :, :] = v_tau[:, :, 1] + vel_vec['y']

    def initialize_lugre(self):
        """
        Internal function to initialize the LuGre variables
        """
        z = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))  # bristles
        f = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))  # tangential force at each grid cell
        dz = np.zeros((2, self.p['grid_shape'][0], self.p['grid_shape'][1]))  # derivative of bristles
        return {'z': z, 'f': f, 'dz': dz}

    def update_lugre(self):
        """
        Updates the LuGre for one time step and calculates a new force at the center of the contact patch.
        :return:
        """
        v_norm = np.linalg.norm(self.velocity_grid, axis=0)
        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * np.exp(- (v_norm / self.p['v_s']) ** self.p['alpha'])
        z_ss = self.steady_state_z(v_norm, g)

        if self.p['steady_state']:
            self.lugre['f'] = (self.p['s0'] * z_ss + self.p[
                's2'] * self.velocity_grid) * self.normal_force_grid
            return

        if self.p['elasto_plastic']:
            beta = self.elasto_plastic(z_ss)
            dz = self.velocity_grid - beta * self.lugre['z'] * (self.p['s0'] * (v_norm / g))
        else:
            dz = self.velocity_grid - self.lugre['z'] * (self.p['s0'] * (v_norm / g))

        if self.p['stability']:
            delta_z = (z_ss - self.lugre['z']) / self.p['dt']
            dz = np.min([abs(dz), abs(delta_z)], axis=0) * np.sign(dz)

        self.lugre['z'] += dz * self.p['dt']

        self.lugre['f'] = (self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz +
                           self.p['s2'] * self.velocity_grid) * self.normal_force_grid
        self.lugre['dz'] = dz

    def steady_state_z(self, v_norm, g):
        """
        Internal function to calculate the steady state bristle deflection.
        :param v_norm:
        :param g:
        :return:
        """
        v_norm1 = v_norm.copy()
        v_norm1[v_norm1 == 0] = 1
        lugre_ss = self.velocity_grid * g / (self.p['s0'] * v_norm1)
        return lugre_ss

    def elasto_plastic(self, z_ss):
        """
        Internal function to
        :param z_ss:
        :return:
        """
        alpha = np.zeros(self.p['grid_shape'])

        for i_x, x_ in enumerate(self.x_pos_vec):
            for i_y, y_ in enumerate(self.y_pos_vec):
                alpha[i_x, i_y] = elasto_plastic_beta(self.lugre['z'][:, i_x, i_y],
                                                       z_ss[:, i_x, i_y],
                                                       self.p['z_ba_ratio'],
                                                       self.velocity_grid[:, i_x, i_y])

        return alpha

    def approximate_integral(self):
        """
        Sums up all the velocity vectors and calculates the frictional torque. Outputs for force wrench at
        the center of the contact patch
        :return:
        """
        fx = - np.sum(self.lugre['f'][0, :, :])
        fy = - np.sum(self.lugre['f'][1, :, :])
        tau = np.cross(self.pos_matrix_2d, self.lugre['f'], axis=0)
        return {'x': fx, 'y': fy, 'tau': -np.sum(tau)}

    def ode_step(self, t, y, vel):
        """
        Test function for simulating with ODE solvers.
        :param t: time
        :param y: state
        :param vel: velocity
        :return: time derivative of y
        """
        shape_ = self.lugre['z'].shape
        self.lugre['z'] = np.reshape(y, shape_)
        vel_vec = {'x': vel[0], 'y': vel[1], 'tau': vel[2]}
        f = self.step_single_point(vel_vec)
        dy = self.lugre['dz']
        return np.reshape(dy, np.prod(shape_)).tolist(), f

    def ode_init(self):
        """
        Initialize for solving with ODE solver.
        :return:
        """
        shape_ = self.lugre['z'].shape
        return np.reshape(self.lugre['z'], np.prod(shape_)).tolist()


class ReducedFrictionModel(FrictionBase):
    """
    Class for simulating the reduced friction model
    """
    def __init__(self, properties: Dict[str, any], nr_ls_segments: int = 20, ls_active: bool = True):
        super().__init__(properties)
        self.ra = 1
        self.limit_surface = CustomHashList3D(nr_ls_segments)
        self.viscous_scale = np.ones(3)
        self.p_x_y = None
        self.distributed_model = self.initialize_distributed_model()
        self.lugre = self.initialize_lugre()
        self.ls_active = ls_active
        self.delta_x = 0
        self.delta_y = 0

    def set_fn(self, fn):
        """
        Updates the normal force of the pressure distribution.
        :param fn: double
        """
        self.p_x_y.set_fn(fn)

    def step(self, vel_vec: Dict[str, float]) -> Dict[str, float]:
        """
        This function does one time step of the friction model
        :param vel_vec: dict with the velocities for the center of the contact patch
        :return: dict with forces at the center of the contact patch
        """
        p, self.cop, self.fn = self.p_x_y.get(self.p['grid_size'])

        vel_cop = vel_to_cop(self.cop, vel_vec)

        self.update_lugre(vel_cop)

        self.force_at_cop = {'x': self.lugre['f'][0, 0], 'y': self.lugre['f'][1, 0], 'tau': self.lugre['f'][2, 0]}

        force = self.move_force_to_center(self.force_at_cop)

        return force

    def initialize_distributed_model(self):
        """
        Internal function to initialize the distributed model which is used to pre-calculate the limit surface
        :return: Object
        """
        properties = copy.deepcopy(self.p)
        properties['mu_c'] = 1
        properties['mu_s'] = 1
        properties['s2'] = 0
        properties['steady_state'] = True
        return DistributedFrictionModel(properties)

    def initialize_lugre(self):
        """
        Internal function to initialize the LuGre parameters
        :return:
        """
        z = np.zeros((3, 1))  # bristles
        f = np.zeros((3, 1))  # tangential force at each grid cell
        return {'z': z, 'f': f}

    def update_p_x_y(self, p_x_y):
        """
        Updates the pressure distribution
        :param p_x_y: Object of type PObject for surfaces.py
        """
        self.p_x_y = p_x_y
        self.distributed_model.update_p_x_y(self.p_x_y)
        self.p['grid_size'] = self.p_x_y.size
        # gamma radius
        self.ra = update_radius(self.distributed_model)
        # viscus scale
        self.viscous_scale = update_viscus_scale(self.distributed_model, self.ra, self.cop)
        # self.update_cop_and_force_grid(p_x_y)

    def update_pre_compute(self):
        """
        Recomputes the limit surface.
        """
        # limit surface
        self.limit_surface.initialize(self.distributed_model)
        self.update_delta_x_y()

    def update_delta_x_y(self):
        """
        Internal function to calculate delta_x and delta_y.
        """
        omega = 1
        vel_cop = {'x':0, 'y':0, 'tau':omega}
        h = self.limit_surface.get_bilinear_interpolation(vel_cop, self.ra)
        vx0 = self.ra*omega*h[0,0]
        vy0 = self.ra*omega*h[1,0]
        vx = vx0
        vy = vy0
        while np.linalg.norm([h[0,0], h[1,0]]) > 1e-6:
            vel_cop = {'x': vx, 'y': vy, 'tau': omega}
            h = self.limit_surface.get_bilinear_interpolation(vel_cop, self.ra)
            if vx0 != 0:
                vx = vx*(self.ra * omega * h[0,0] + vx0)/vx0
            if vy0 != 0:
                vy = vy*(self.ra * omega * h[1,0] + vy0)/vy0
        self.delta_x = -vy/omega
        self.delta_y = vx / omega

    def update_lugre(self, vel_cop):
        """
        Updates the LuyGre model
        :param vel_cop: The velocity wrench at the CoP
        :return:
        """
        vel_cop_list = np.array([[vel_cop['x']], [vel_cop['y']], [vel_cop['tau']]])
        S = np.diag([1, 1, self.ra])

        A = np.eye(3)
        A[2, 2] = self.ra**2
        sx, sy = self.calc_skew_variables(vel_cop)
        A[0,2] = sx
        A[1,2] = sy
        A_sym = (A.T + A)/2
        v_n = np.sqrt(vel_cop_list.T.dot(A_sym).dot(vel_cop_list))

        if self.ls_active:
            h = self.limit_surface.get_bilinear_interpolation(vel_cop, self.ra)

            w_vn = np.sign(np.diag(A.dot(vel_cop_list).flatten())).dot(S).dot(np.abs(h))*v_n
        else:
            w_vn = A.dot(vel_cop_list)

        g = self.p['mu_c'] + (self.p['mu_s'] - self.p['mu_c']) * np.exp(- (v_n / self.p['v_s']) ** self.p['alpha'])

        if v_n != 0:
            z_ss = (w_vn * g) / (self.p['s0'] * v_n)
        else:
            z_ss = np.zeros((3, 1))

        if self.p['steady_state']:
            self.lugre['f'] = -(self.p['s0'] * z_ss + self.viscous_scale * self.p['s2'] * A.dot(vel_cop_list)) * self.fn
            return

        if self.p['elasto_plastic']:
            beta = elasto_plastic_beta(np.linalg.pinv(S).dot(self.lugre['z']).flatten(),
                                         np.linalg.pinv(S).dot(z_ss).flatten(),
                                         self.p['z_ba_ratio'],
                                         np.linalg.pinv(S).dot(w_vn).flatten())

        else:
            beta = 1

        dz = w_vn - beta * self.lugre['z'] * (self.p['s0'] * (v_n / g))

        if self.p['stability']:
            z = self.lugre['z']
            delta_z = (z_ss - z) / self.p['dt']

            dz = np.min([abs(dz), abs(delta_z)], axis=0) * np.sign(dz)
        self.lugre['z'] += dz * self.p['dt']

        f = -(self.p['s0'] * self.lugre['z'] + self.p['s1'] * dz +
              self.viscous_scale * self.p['s2'] * A.dot(vel_cop_list)) * self.fn
        self.lugre['f'] = f
        self.lugre['dz'] = dz

    def calc_skew_variables(self, vel_cop):
        """
        Calculates the skew variables based on delta_x and delta_y
        :param vel_cop: The velocity wrench at the CoP
        :return: sx and sy
        """
        ras = self.limit_surface.ra
        p_xy_n = np.linalg.norm([self.delta_x,  self.delta_y])
        if p_xy_n != 0:
            p_x_n = abs(self.delta_x) / p_xy_n
            p_y_n = abs(self.delta_y) / p_xy_n
        else:
            p_x_n = 0
            p_y_n = 0
        nn = np.linalg.norm([vel_cop['x'] * p_x_n, vel_cop['y'] * p_y_n])
        s_n = (2*np.arctan2(self.ra * abs(vel_cop['tau']), nn))/np.pi
        sx = -self.ra/ras * s_n * self.delta_y
        sy = self.ra/ras * s_n * self.delta_x
        return sx, sy


    def ode_step(self, t, y, vel):
        """
        Test function for simulating with ODE solvers.
        :param t: time
        :param y: state
        :param vel: velocity
        :return: time derivative of y
        """
        shape_ = self.lugre['z'].shape
        self.lugre['z'] = np.reshape(y, shape_)
        vel_vec = {'x': vel[0], 'y': vel[1], 'tau': vel[2]}
        f = self.step(vel_vec)
        dy = self.lugre['dz']
        return np.reshape(dy, np.prod(shape_)).tolist(), f

    def ode_init(self):
        """
        Initialize for solving with ODE solver.
        :return:
        """
        shape_ = self.lugre['z'].shape
        return np.reshape(self.lugre['z'], np.prod(shape_)).tolist()

