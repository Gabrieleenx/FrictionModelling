import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
from utils import *
from layout import get_layout
from surfaces.surfaces import p_square, p_line, p_circle
from frictionModels.frictionModel import FullFrictionModel, ReducedFrictionModel

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

shape_set = {'Square': p_square, 'Circle': p_circle, 'Line': p_line}

app.layout = get_layout()

@app.callback(
    Output('store_surface_data', 'data'),
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
def update_surface_data(model,
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

    properties = {'grid_shape': (grid_shape, grid_shape),  # number of grid elements in x any
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

    data = dict()

    if 'Full' in model:
        planar_lugre = FullFrictionModel(properties=properties)
        planar_lugre.update_p_x_y(shape_set[shape])
        f_surf, t_surf, x_y = calc_surface(planar_lugre, lin_vel_range, ang_vel_range, res)
        data_local = dict()
        data_local['f'] = f_surf
        data_local['tau'] = t_surf
        data_local['x'] = x_y[0]
        data_local['y'] = x_y[1]
        data['Full'] = data_local

    if 'Reduced with LS' in model:
        planar_lugre_reduced = ReducedFrictionModel(properties=properties)
        planar_lugre_reduced.update_p_x_y(shape_set[shape])
        planar_lugre_reduced.update_pre_compute()
        f_surf, t_surf, x_y = calc_surface(planar_lugre_reduced, lin_vel_range, ang_vel_range, res)
        data_local = dict()
        data_local['f'] = f_surf
        data_local['tau'] = t_surf
        data_local['x'] = x_y[0]
        data_local['y'] = x_y[1]
        data['Reduced with LS'] = data_local

    if 'Reduced with ellipse' in model:
        planar_lugre_reduced_ellipse = ReducedFrictionModel(properties=properties, ls_active=False)
        planar_lugre_reduced_ellipse.update_p_x_y(shape_set[shape])
        planar_lugre_reduced_ellipse.update_pre_compute()
        f_surf, t_surf, x_y = calc_surface(planar_lugre_reduced_ellipse, lin_vel_range, ang_vel_range, res)
        data_local = dict()
        data_local['f'] = f_surf
        data_local['tau'] = t_surf
        data_local['x'] = x_y[0]
        data_local['y'] = x_y[1]
        data['Reduced with ellipse'] = data_local

    return data


@app.callback([Output('3d_force', 'figure'),
               Output('3d_force', 'config')],
              Input('store_surface_data', 'data'),
              Input('format_dropdown', 'value'))
def update_fig_force(data, format_value):
    format_value = format_value if format_value is not None else 'png'

    fig = go.Figure()

    for key, value in data.items():
        fig.add_trace(go.Surface(z=value['f'], x=value['x'], y=value['y']))

    fig.update_layout(title='Force', autosize=False,
                      width=800, height=800,
                      scene=dict(
                          xaxis_title='Lin vel',
                          yaxis_title='Ang vel',
                          zaxis_title='Force'),
                      margin=dict(l=0, r=0, b=0, t=30),
                      uirevision=True)

    config = {'toImageButtonOptions': {'format': format_value}}

    return [fig, config]

@app.callback([Output('3d_torque', 'figure'),
               Output('3d_torque', 'config')],
              Input('store_surface_data', 'data'),
              Input('format_dropdown', 'value'))
def update_fig_force(data, format_value):
    format_value = format_value if format_value is not None else 'png'

    fig = go.Figure()

    for key, value in data.items():
        fig.add_trace(go.Surface(z=value['tau'], x=value['x'], y=value['y']))

    fig.update_layout(title='Torque', autosize=False,
                      width=800, height=800,
                      scene=dict(
                          xaxis_title='Lin vel',
                          yaxis_title='Ang vel',
                          zaxis_title='Torque'),
                      margin=dict(l=0, r=0, b=0, t=30),
                      uirevision=True)

    config = {'toImageButtonOptions': {'format': format_value}}

    return [fig, config]


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
                  'dt': 1e-4,
                  'stability': True,
                  'elasto_plastic': True,
                  'z_ba_ratio': 0.9,
                  'steady_state': True}

    planar_lugre = FullFrictionModel(properties=properties)
    planar_lugre.update_p_x_y(shape_set[shape])
    planar_lugre.step(vel_vec={'x': lin_vel, 'y': 0, 'tau': ang_vel})
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

    add_lines(x_vec, y_vec, grid_size, fig)
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
    Input('ratio', 'value')
)
def update_ellipse(shape,
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
                   ratio):

    properties = {'grid_shape': (grid_shape, grid_shape),  # number of grid elements in x any
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
    planar_lugre.update_p_x_y(shape_set[shape])
    gamma = 0.008
    num = 200
    ang_vel_list = np.linspace(ang_vel, 0, num)
    lin_vel_list = np.linspace(0, lin_vel, num)
    fig = go.Figure()

    data = np.zeros((2, num))
    for i in range(num):
        f = planar_lugre.step(vel_vec={'x': lin_vel_list[i], 'y': 0, 'tau': ang_vel_list[i]})
        data[0, i] = np.linalg.norm([f['x'], f['y']])
        data[1, i] = abs(f['tau'])
    max_save = np.max(data, axis=1)
    data = data.T / max_save
    data = data.T
    fig.add_trace(go.Scatter(x=data[0, :], y=data[1, :], name=str('Limit surface square')))

    x = np.linspace(0,1,100)
    y = np.sqrt(1 - x**2)
    fig.add_trace(go.Scatter(x=x, y=y, name='ellipse'))

    arrows = []
    vx = ratio*lin_vel
    vt = (1-ratio)*ang_vel
    f = planar_lugre.step(vel_vec={'x': vx, 'y': 0, 'tau': vt})
    dx = np.linalg.norm([f['x'], f['y']]) / max_save[0]
    dy = abs(f['tau']) / max_save[1]

    arrows += [return_arrow(0, 0, dx, dy)]

    dx = vx / np.linalg.norm([vx, gamma*vt])
    dy = gamma*vt / np.linalg.norm([vx, gamma*vt])

    arrows += [return_arrow(0, 0, dx, dy)]
    fig.update_layout(annotations=arrows)


    return fig

if __name__ == '__main__':
    app.run_server(debug=True)