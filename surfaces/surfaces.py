import numpy as np
from surfaces.utils import create_circular_mask


class PObject(object):
    def __init__(self, size, shape_, shape_fun, fn=1):
        self.size = size
        self.shape_fun = shape_fun
        self.shape_ = shape_
        self.fn = fn
        self.cop = np.zeros(2)
        self.set_pressure_shape(size, shape_, shape_fun, fn)

    def set_fn(self, fn):
        self.fn = fn

    def get(self, size):
        # TODO: return the shape and size.
        p = self.pressure_grid_norm*self.fn
        cop = self.cop*size/self.size
        return p, cop, self.fn

    def set_pressure_shape(self, size, shape_, shape_fun, fn=1):
        # TODO: the actual shape
        area = size**2
        pressure_grid = shape_fun(shape_) * area
        fn_ = np.sum(pressure_grid)

        self.size = size
        self.pressure_grid_norm = pressure_grid/fn_
        self.fn = fn

        x_pos = (np.arange(shape_[0]) + 0.5 - shape_[0] / 2) * size
        y_pos = (np.arange(shape_[1]) + 0.5 - shape_[1] / 2) * size

        self.cop[0] = np.sum(x_pos.dot(self.pressure_grid_norm))
        self.cop[1] = np.sum(y_pos.dot(self.pressure_grid_norm.T))

def p_square(shape_):
    return np.ones(shape_) * 1e3


def p_circle(shape_):
    m = np.ones(shape_) * 1e3
    return m * create_circular_mask(shape_[0], shape_[1])


def p_line(shape_):
    shape = shape_
    p = np.zeros(shape)
    if shape[0]/2 == shape[0]//2:
        p[shape[0]//2-1, :] = np.ones(shape[0]) * 1e3
        p[shape[0]//2, :] = np.ones(shape[0]) * 1e3
        p = p/2
    else:
        p[shape[0]//2, :] = np.ones(shape[0]) * 1e3
    return p


def p_line_grad(shape_):
    shape = shape_
    p = np.zeros(shape)
    if shape[0]/2 == shape[0]//2:
        p[shape[0]//2-1, :] = np.arange(shape[0]) * 1e3
        p[shape[0]//2, :] = np.arange(shape[0]) * 1e3
        p = p/2
    else:
        p[shape[0]//2, :] = np.arange(shape[0]) * 1e3
    return p
