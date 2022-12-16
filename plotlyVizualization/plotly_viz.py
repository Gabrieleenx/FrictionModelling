import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
from utils import compute_horizontal_lines, compute_vertical_lines
from layout import get_layout
import sys
sys.path.append('../frictionModeling')
from surfaces.surfaces import p_square, p_line, p_circle
from frictionModels.frictionModel import FullFrictionModel, ReducedFrictionModel

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

shape_set = {'Square': p_square, 'Circle': p_circle, 'Line': p_line}

app.layout = get_layout()

@app.callback(
    Output('3d_force', 'figure'),
    Output('3d_torque', 'figure'),
    Input('friction_model', 'value'),
    Input('shape', 'value'),
    Input('linear_vel_range', 'value'),
    Input('angular_vel_range', 'value'),
    Input('grid_shape', 'value'),
    Input('grid_size', 'value'),
    Input('mu_c', 'value'),
    Input('mu_s', 'value'),
    Input('v_s', 'value'),
    Input('alpha', 'value'),
    Input('s0', 'value'),
    Input('s1', 'value'),
    Input('s2', 'value'),
    Input('resolution', 'value'),
)
def update_3d_force(model,
                    shape,
                    lin_vel_range,
                    ang_vel_range,
                    grid_shape,
                    grid_size,
                    mu_c,
                    mu_s,
                    v_s,
                    alpha,
                    s0,
                    s1,
                    s2,
                    res):

    properties = {'grid_shape': (grid_shape[0], grid_shape[1]),  # number of grid elements in x any
                  'grid_size': grid_size,  # the physical size of each grid element
                  'mu_c': mu_c,
                  'mu_s': mu_s,
                  'v_s': v_s,
                  'alpha': alpha,
                  's0': s0,
                  's1': s1,
                  's2': s2,
                  'dt': 1e-4,
                  'stability': True,
                  'elasto_plastic': True,
                  'z_ba_ratio': 0.9,
                  'steady_state': True}

    planar_lugre = FullFrictionModel(properties=properties)
    planar_lugre_reduced = ReducedFrictionModel(properties=properties)
    planar_lugre_reduced_ellipse = ReducedFrictionModel(properties=properties, ls_active=False)

    planar_lugre.update_p_x_y(shape_set[shape])

    planar_lugre_reduced.update_p_x_y(shape_set[shape])
    planar_lugre_reduced.update_pre_compute()

    planar_lugre_reduced_ellipse.update_p_x_y(shape_set[shape])
    planar_lugre_reduced_ellipse.update_pre_compute()

    x = np.linspace(lin_vel_range[0], lin_vel_range[1], res)
    y = np.linspace(ang_vel_range[0], ang_vel_range[1], res)

    f_surf = np.zeros((res, res))
    t_surf = np.zeros((res, res))

    f_surf_red = np.zeros((res, res))
    t_surf_red = np.zeros((res, res))

    f_surf_red_ellipse = np.zeros((res, res))
    t_surf_red_ellipse = np.zeros((res, res))

    for x_, v in enumerate(x):
        for y_, w in enumerate(y):
            if 'Full' in model:
                f = planar_lugre.step(vel_vec={'x':v, 'y': 0, 'tau': w})
                f_surf[x_, y_] = np.linalg.norm([f['x'], f['y']])
                t_surf[x_, y_] = abs(f['tau'])

            if 'Reduced' in model:
                f = planar_lugre_reduced.step(vel_vec={'x': v, 'y': 0, 'tau': w})
                f_surf_red[x_, y_] = np.linalg.norm([f['x'], f['y']])
                t_surf_red[x_, y_] = abs(f['tau'])

            if 'Reduced ellipse' in model:
                f = planar_lugre_reduced_ellipse.step(vel_vec={'x': v, 'y': 0, 'tau': w})
                f_surf_red_ellipse[x_, y_] = np.linalg.norm([f['x'], f['y']])
                t_surf_red_ellipse[x_, y_] = abs(f['tau'])

    fig_force = go.Figure()
    fig_torque = go.Figure()

    if 'Full' in model:
        fig_force.add_trace(go.Surface(z=f_surf.T, x=x, y=y))
        fig_torque.add_trace(go.Surface(z=t_surf.T, x=x, y=y))

    if 'Reduced' in model:
        fig_force.add_trace(go.Surface(z=f_surf_red.T, x=x, y=y, opacity=0.7))
        fig_torque.add_trace(go.Surface(z=t_surf_red.T, x=x, y=y, opacity=0.7))

    if 'Reduced ellipse' in model:
        fig_force.add_trace(go.Surface(z=f_surf_red_ellipse.T, x=x, y=y, opacity=0.7))
        fig_torque.add_trace(go.Surface(z=t_surf_red_ellipse.T, x=x, y=y, opacity=0.7))


    fig_force.update_layout(title='Linear force', autosize=False,
                      width=500, height=500,
                      scene=dict(
                          xaxis_title='lin vel',
                          yaxis_title='ang vel',
                          zaxis_title='lin force'),
                      margin=dict(l=0, r=0, b=0, t=30),
                      uirevision=True)

    fig_torque.update_layout(title='Torque', autosize=False,
                      width=500, height=500,
                      scene=dict(
                          xaxis_title='lin vel',
                          yaxis_title='ang vel',
                          zaxis_title='torque'),
                      margin=dict(l=0, r=0, b=0, t=30),
                      uirevision=True)

    return fig_force, fig_torque


@app.callback(
    Output('contact_surface_plt', 'figure'),
    Input('shape', 'value'),
    Input('linear_vel_value', 'value'),
    Input('angular_vel_value', 'value'),
    Input('grid_shape', 'value'),
    Input('grid_size', 'value'),
    Input('mu_c', 'value'),
    Input('mu_s', 'value'),
    Input('v_s', 'value'),
    Input('alpha', 'value'),
    Input('s0', 'value'),
    Input('s1', 'value'),
    Input('s2', 'value'),
    Input('contact_surface', 'value')
)
def update_contact(shape,
                  lin_vel,
                  ang_vel,
                  grid_shape,
                  grid_size,
                  mu_c,
                  mu_s,
                  v_s,
                  alpha,
                  s0,
                  s1,
                  s2,
                  contact_surface):

    properties = {'grid_shape': (grid_shape, grid_shape),  # number of grid elements in x any
                  'grid_size': grid_size,  # the physical size of each grid element
                  'mu_c': mu_c,
                  'mu_s': mu_s,
                  'v_s': v_s,
                  'alpha': alpha,
                  's0': s0,
                  's1': s1,
                  's2': s2,
                  'dt': 1e-4}

    planar_lugre = PlanarFriction(properties=properties)
    f = planar_lugre.steady_state(vel_vec={'x': lin_vel, 'y': 0, 'tau': ang_vel},
                                  p_x_y=shape_set[shape])

    x_vec = planar_lugre.x_pos_vec
    y_vec = planar_lugre.y_pos_vec
    normal_f = planar_lugre.normal_force_grid

    fig = go.Figure()

    if 'Pressure' in contact_surface:
        fig.add_trace(go.Heatmap(
            z=normal_f.T,
            x=x_vec,
            y=y_vec
        ))
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    x_vec_ = (x_vec - grid_size/2)
    x_vec_ = np.append(x_vec_, np.max(x_vec)+grid_size)
    y_vec_ = (y_vec - grid_size/2)
    y_vec_ = np.append(y_vec_, np.max(y_vec)+grid_size)

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

    if 'Vel' in contact_surface:
        arrows = []
        scale_arrow = 0.05
        for ix, x_ in enumerate(x_vec):
            for iy, y_ in enumerate(y_vec):
                dx = planar_lugre.velocity_grid[0, ix, iy]
                dy = planar_lugre.velocity_grid[1, ix, iy]
                arrow = {'x':x_+dx*scale_arrow,  # arrows' head
                    'y':y_+dy*scale_arrow,  # arrows' head
                    'ax':x_,  # arrows' tail
                    'ay':y_,  # arrows' tail
                    'xref':'x',
                    'yref':'y',
                    'axref':'x',
                    'ayref':'y',
                    'text':'',  # if you want only the arrow
                    'showarrow':True,
                    'arrowhead':3,
                    'arrowsize':1,
                    'arrowwidth':1,
                    'arrowcolor':'black'
                }
                arrows += [arrow]
        fig.update_layout(annotations=arrows)

    return fig


@app.callback(
    Output('ellipsoid', 'figure'),
    Input('linear_vel_value', 'value'),
    Input('angular_vel_value', 'value'),
    Input('grid_shape', 'value'),
    Input('grid_size', 'value'),
    Input('mu_c', 'value'),
    Input('mu_s', 'value'),
    Input('v_s', 'value'),
    Input('alpha', 'value'),
    Input('s0', 'value'),
    Input('s1', 'value'),
    Input('s2', 'value'),
)
def update_ellipse(lin_vel,
                   ang_vel,
                   grid_shape,
                   grid_size,
                   mu_c,
                   mu_s,
                   v_s,
                   alpha,
                   s0,
                   s1,
                   s2):

    properties = {'grid_shape': (grid_shape, grid_shape),  # number of grid elements in x any
                  'grid_size': grid_size,  # the physical size of each grid element
                  'mu_c': mu_c,
                  'mu_s': mu_s,
                  'v_s': v_s,
                  'alpha': alpha,
                  's0': s0,
                  's1': s1,
                  's2': s2,
                  'dt': 1e-4}

    planar_lugre = PlanarFriction(properties=properties)
    num = 200
    ang_vel_list = np.linspace(ang_vel, 0, num)
    lin_vel_list = np.linspace(0, lin_vel, num)
    fig = go.Figure()
    """
    for shape_ in shape_set:
        if shape_ == 'Line':
            continue
        data = np.zeros((2, num))
        for i in range(num):
            f = planar_lugre.steady_state(vel_vec={'x':lin_vel_list[i], 'y': 0, 'tau': ang_vel_list[i]}, p_x_y=shape_set[shape_])
            data[0, i] = np.linalg.norm([f['x'], f['y']])
            data[1, i] = abs(f['tau'])
        data = data.T/np.max(data, axis=1)
        data = data.T
        fig.add_trace(go.Scatter(x=data[0, :], y=data[1, :], name=str(shape_)))

    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                      xaxis_title="Force",
                      yaxis_title="Torque",
                      margin=dict(l=10, r=10, b=30, t=30),
                      uirevision=True)
    """

    data = np.zeros((2, num))
    for i in range(num):
        f = planar_lugre.steady_state(vel_vec={'x': lin_vel_list[i], 'y': 0, 'tau': ang_vel_list[i]},
                                      p_x_y=shape_set['Square'])
        data[0, i] = np.linalg.norm([f['x'], f['y']])
        data[1, i] = abs(f['tau'])
    max_save =  np.max(data, axis=1)
    data = data.T / max_save
    data = data.T
    fig.add_trace(go.Scatter(x=data[0, :], y=data[1, :], name=str('Limit surface square')))


    x = np.linspace(0,1,100)
    y = np.sqrt(1 - x**2)
    fig.add_trace(go.Scatter(x=x, y=y, name='ellipse'))






    arrows = []
    f = planar_lugre.steady_state(vel_vec={'x': lin_vel, 'y': 0, 'tau': ang_vel},
                                  p_x_y=shape_set['Square'])
    dx = np.linalg.norm([f['x'], f['y']]) / max_save[0]
    dy = abs(f['tau']) / max_save[1]


    arrow = {'x': 0 + dx,  # arrows' head
             'y': 0 + dy,  # arrows' head
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
             'arrowcolor': 'black'
             }
    arrows += [arrow]

    dx = lin_vel / np.linalg.norm([lin_vel, 0.008*ang_vel])
    dy = 0.008*ang_vel / np.linalg.norm([lin_vel, 0.008*ang_vel])

    arrow = {'x': 0 + dx,  # arrows' head
             'y': 0 + dy,  # arrows' head
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
             'arrowcolor': 'black'
             }
    arrows += [arrow]
    fig.update_layout(annotations=arrows)


    return fig

if __name__ == '__main__':
    app.run_server(debug=True)