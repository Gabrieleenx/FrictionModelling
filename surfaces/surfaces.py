import numpy as np
from surfaces.utils import create_circular_mask

def p_square(M):
    return np.ones(M[1, :, :].shape) * 1e3

def p_circle(M):
    m = np.ones(M[1, :, :].shape) * 1e3
    return m * create_circular_mask(M[1, :, :].shape[0], M[1, :, :].shape[1])

def p_line(M):
    shape = M[1, :, :].shape
    p = np.zeros(shape)
    if shape[0]/2 == shape[0]//2:
        p[shape[0]//2-1, :] = np.ones(shape[0]) * 1e3
        p[shape[0]//2, :] = np.ones(shape[0]) * 1e3
        p = p/2
    else:
        p[shape[0]//2, :] = np.ones(shape[0]) * 1e3
    return p
