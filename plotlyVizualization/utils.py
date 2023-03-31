import numpy as np
import plotly.graph_objects as go

def compute_horizontal_lines(x_min, x_max, y_data):
    x = np.tile([x_min, x_max, None], len(y_data))
    y = np.ndarray.flatten(np.array([[a, a, None] for a in y_data]))
    return x, y

def compute_vertical_lines(y_min, y_max, x_data):
    y = np.tile([y_min, y_max, None], len(x_data))
    x = np.ndarray.flatten(np.array([[a, a, None] for a in x_data]))
    return x, y

def add_lines(x_vec, y_vec, grid_size, fig):
    x_vec_ = (x_vec - grid_size/2)
    x_vec_ = np.append(x_vec_, np.max(x_vec)+grid_size/2)
    y_vec_ = (y_vec - grid_size/2)
    y_vec_ = np.append(y_vec_, np.max(y_vec)+grid_size/2)

    hx, hy = compute_horizontal_lines(np.min(x_vec_), np.max(x_vec_), y_vec_)
    fig.add_trace(go.Scatter(
        x=hx,
        y=hy,
        opacity=0.5,
        marker_color='white',
        line_width=1,
    ))

    vx, vy = compute_vertical_lines(np.min(y_vec_), np.max(y_vec_), x_vec_)
    fig.add_trace(go.Scatter(
        x=vx,
        y=vy,
        opacity=0.5,
        marker_color='white',
        line_width=1,
    ))


def calc_surface(model, lin_vel_range, ang_vel_range, res, direction):
    x = np.linspace(lin_vel_range[0], lin_vel_range[1], res)
    y = np.linspace(ang_vel_range[0], ang_vel_range[1], res)

    f_surf = np.zeros((res, res))
    t_surf = np.zeros((res, res))

    for x_, v in enumerate(x):
        for y_, w in enumerate(y):
            vx = np.cos(direction) * v
            vy = np.sin(direction) * v
            f = model.step(vel_vec={'x': vx, 'y': vy, 'tau': w})
            f = model.force_at_cop
            f_surf[x_, y_] = np.linalg.norm([f['x'], f['y']])
            t_surf[x_, y_] = abs(f['tau'])

    return f_surf.T, t_surf.T, [x, y]

def return_arrow(sx, sy, dx, dy, color='black'):
    arrow = {'x': sx + dx,  # arrows' head
             'y': sy + dy,  # arrows' head
             'ax': 0,  # arrows' tail
             'ay': 0,  # arrows' tail
             'xref': 'x',
             'yref': 'y',
             'axref': 'x',
             'ayref': 'y',
             'text': '',  # if you want only the arrow
             'showarrow': True,
             'arrowhead': 3,
             'arrowsize': 1,
             'arrowwidth': 1,
             'arrowcolor': color
             }
    return arrow