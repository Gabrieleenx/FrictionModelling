import numpy as np
from surfaces.utils import create_circular_mask


class PObject(object):
    def __init__(self, size, shape_, shape_fun, fn=1):
        """
        Object for the pressure contact
        :param size: Size in [m] for a grid cell
        :param shape_: (int, int) how many cells
        :param shape_fun: the function that returns the normal force of each cell
        :param fn: The total normal force
        """
        self.size = size
        self.shape_fun = shape_fun
        self.shape_ = shape_
        self.fn = fn
        self.cop = np.zeros(2)
        self.set_pressure_shape(size, shape_, shape_fun, fn)

    def set_fn(self, fn):
        """
        Change the normal force
        :param fn: double
        """
        self.fn = fn

    def get(self, size):
        """
        Returns the properties of the contact surface
        :param size: size of a cell
        :return: normal force for each cell, location of CoP and the total normal force.
        """
        p = self.pressure_grid_norm*self.fn
        cop = self.cop*size/self.size
        return p, cop, self.fn

    def set_pressure_shape(self, size, shape_, shape_fun, fn=1):
        """
        Update the normalized pressure distribution
        :param size: double
        :param shape_: (int, int)
        :param shape_fun: the function that returns the normal force of each cell
        :param fn: fn=1
        :return:
        """
        area = size**2
        pressure_grid = shape_fun(shape_, fn) * area
        fn_ = np.sum(pressure_grid)

        self.size = size
        self.pressure_grid_norm = pressure_grid/fn_
        self.fn = fn

        x_pos = (np.arange(shape_[0]) + 0.5 - shape_[0] / 2) * size
        y_pos = -(np.arange(shape_[1]) + 0.5 - shape_[1] / 2) * size

        self.cop[0] = np.sum(x_pos.dot(self.pressure_grid_norm.T))
        self.cop[1] = np.sum(y_pos.dot(self.pressure_grid_norm))

def p_square(shape_, fn):
    """
    A function that returns the normal force of each cell
    :param shape_: (int, int)
    :return:
    """
    return np.ones(shape_) * 1e3

def p_circle(shape_, fn):
    """
    A function that returns the normal force of each cell
    :param shape_: (int, int)
    :return:
    """
    m = np.ones(shape_) * 1e3
    return m * create_circular_mask(shape_[0], shape_[1])

def p_line(shape_, fn):
    """
    A function that returns the normal force of each cell
    :param shape_: (int, int)
    :return:
    """
    shape = shape_
    p = np.zeros(shape)
    if shape[0]/2 == shape[0]//2:
        p[:, shape[0]//2-1] = np.ones(shape[0]) * 1e3
        p[:, shape[0]//2] = np.ones(shape[0]) * 1e3
        p = p/2
    else:
        p[:, shape[0]//2] = np.ones(shape[0]) * 1e3
    return p

def p_line_grad3(shape_, fn):
    """
    A function that returns the normal force of each cell
    :param shape_: (int, int)
    :return:
    """
    shape = shape_
    p = np.zeros(shape)
    if shape[0]/2 == shape[0]//2:
        p[shape[0]//2-1, :] = (shape[0] - np.arange(shape[0])) * 1e3
        p[shape[0]//2, :] = (shape[0] - np.arange(shape[0])) * 1e3
        p = p/2
    else:
        p[shape[0]//2, :] = (shape[0] - np.arange(shape[0])) * 1e3
    return p


def p_line_grad2(shape_, fn):
    """
    A function that returns the normal force of each cell
    :param shape_: (int, int)
    :return:
    """
    shape = shape_
    p = np.zeros(shape)
    if shape[0]/2 == shape[0]//2:
        p[shape[0]//2-1, :] = (np.arange(shape[0])) * 1e3
        p[shape[0]//2, :] = (np.arange(shape[0])) * 1e3
        p = p/2
    else:
        p[shape[0]//2, :] = (np.arange(shape[0])) * 1e3
    return p

def p_line_grad(shape_, fn):
    """
    A function that returns the normal force of each cell
    :param shape_: (int, int)
    :return:
    """
    shape = shape_
    p = np.zeros(shape)
    if shape[1]/2 == shape[1]//2:
        p[:, shape[1]//2-1] = (shape[1] - np.arange(shape[1])) * 1e3
        p[:, shape[1]//2] = (shape[1] - np.arange(shape[1])) * 1e3
        p = p/2
    else:
        p[:, shape[1]//2] = (shape[1] - np.arange(shape[1])) * 1e3
    return p

def p_line_grad4(shape_, fn):
    """
    A function that returns the normal force of each cell
    :param shape_: (int, int)
    :return:
    """
    shape = shape_
    p = np.zeros(shape)
    if shape[1]/2 == shape[1]//2:
        p[:, shape[1]//2-1] = (np.arange(shape[1])) * 1e3
        p[:, shape[1]//2] = (np.arange(shape[1])) * 1e3
        p = p/2
    else:
        p[:, shape[1]//2] = (np.arange(shape[1])) * 1e3
    return p


def non_convex_1(shape, fn):
    p = np.zeros(shape)
    n = int(0.15*shape[0])
    y1 = round(shape[0]/2 - n/2)
    y2 = round(shape[0] / 2 + n / 2)
    p[y1:y2,0:n] = 1
    p[y1:y2,shape[0]-n:shape[0]] = 1
    return p


def non_convex_2(shape, fn):
    p = np.zeros(shape)
    n = int(0.15*shape[0])
    n1 = int(0.70*shape[0])
    y1 = round(shape[0]/2 - n/2)
    y2 = round(shape[0] / 2 + n / 2)
    y3 = round(shape[0]/2 - n1/2)
    y4 = round(shape[0] / 2 + n1 / 2)
    p[y1:y2,0:n1] = 1
    p[y3:y4,shape[0]-n:shape[0]] = 1
    return p


def proportional_surface_circle(shape, fn):
    nx = shape[0]
    ny = shape[1]
    cx = (nx-1)/2.0
    cy = (ny-1)/ 2.0
    mu = 1/3
    #mu = 0.5 # silicone
    c = 6 # originally 2.07 in Modeling of Contact Mechanics and Friction Limit Surfaces for Soft Fingers in Robotics, with Experimental Results
    c_k = 1.144 # the effects of this will be normalized away later.
    k = 1/0.5
    a_max = cx +0.5
    a = c*fn**mu
    p = np.zeros(shape)
    if a > a_max:
        a = a_max

    for ix in range(nx):
        for iy in range(ny):
            p_x = (ix-cx)
            p_y = (iy-cy)
            r = np.linalg.norm([p_x, p_y])
            p[iy, ix] = p_r(r, a, k, c_k, fn)
    return p

def proportional_surface_circle2(shape, fn):
    nx = shape[0]
    ny = shape[1]
    cx = (nx-1)/2.0
    cy = (ny-1)/ 2.0
    mu = 1/3
    #mu = 0.5 # silicone
    c = 6 # originally 2.07 in Modeling of Contact Mechanics and Friction Limit Surfaces for Soft Fingers in Robotics, with Experimental Results
    c_k = 1.144 # the effects of this will be normalized away later.
    k = fn/0.5
    a_max = cx +0.5
    a = c*fn**mu
    p = np.zeros(shape)
    if a > a_max:
        a = a_max

    for ix in range(nx):
        for iy in range(ny):
            p_x = (ix-cx)
            p_y = (iy-cy)
            r = np.linalg.norm([p_x, p_y])
            p[iy, ix] = p_r(r, a, k, c_k, fn)
    return p


"""
def proportional_surface_circle2(shape, fn):
    nx = shape[0]
    ny = shape[1]
    cx = (nx-1)/2.0
    cy = (ny-1)/ 2.0
    mu = 0.2590 # silicone
    mu = 0.5 # silicone

    c = 6 # originally 2.07 in Modeling of Contact Mechanics and Friction Limit Surfaces for Soft Fingers in Robotics, with Experimental Results
    c_k = 1.144 # the effects of this will be normalized away later.
    k = fn/mu
    fn_max = 4
    a_max = c*fn_max**0.2590
    a_scale = a_max/cx
    a = c*fn**mu
    p = np.zeros(shape)
    if fn > fn_max:
        a = a_max

    for ix in range(nx):
        for iy in range(ny):
            p_x = (ix-cx)*a_scale
            p_y = (iy-cy)*a_scale
            r = np.linalg.norm([p_x, p_y])
            p[iy, ix] = p_r(r, a, k, c_k, fn)
    return p
"""
def p_r(r, a, k, c_k, fn):
    if r > a:
        return 0
    return c_k * fn/(np.pi * a**2) * (1 - (r/a)**k)**(1/k)

def p_r2(r, a, k, c_k, fn):
    if r > a:
        return 0
    return fn*((fn-0.6)/(r/8+(fn-0.6)))


