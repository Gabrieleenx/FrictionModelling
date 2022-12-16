import numpy as np

def compute_horizontal_lines(x_min, x_max, y_data):
    x = np.tile([x_min, x_max, None], len(y_data))
    y = np.ndarray.flatten(np.array([[a, a, None] for a in y_data]))
    return x, y

def compute_vertical_lines(y_min, y_max, x_data):
    y = np.tile([y_min, y_max, None], len(x_data))
    x = np.ndarray.flatten(np.array([[a, a, None] for a in x_data]))
    return x, y
