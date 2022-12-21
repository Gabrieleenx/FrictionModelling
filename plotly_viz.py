import numpy as np
from dash import Dash, html, dcc, Input, Output
from plotly.subplots import make_subplots

import plotly.graph_objects as go
from friction import PlanarFriction
from friction_simple import PlanarFrictionReduced
from pre_compute_ls import CustomHashList

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)


def p_square(M):
    return np.ones(M[1, :, :].shape) * 1e3


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


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

shape_set = {'Square': p_square, 'Circle': p_circle, 'Line': p_line}


def compute_horizontal_lines(x_min, x_max, y_data):
    x = np.tile([x_min, x_max, None], len(y_data))
    y = np.ndarray.flatten(np.array([[a, a, None] for a in y_data]))
    return x, y

def compute_vertical_lines(y_min, y_max, x_data):
    y = np.tile([y_min, y_max, None], len(x_data))
    x = np.ndarray.flatten(np.array([[a, a, None] for a in x_data]))
    return x, y


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label('Friction Model'),
            dcc.Checklist(
                ['Full', 'Reduced', 'Reduced + gamma', 'Gamma_max', 'Ellipse comp'],
                ['Full'],
                id='friction_model',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            html.Label('Contact Surface'),
            dcc.Checklist(
                ['Pressure', 'Vel', 'COP', 'COR', 'Gamma'],
                ['Pressure'],
                id='contact_surface',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )

        ], style={'width': '24%', 'float': 'right', 'display': 'inline-block'}),
        html.Div([

            html.Label('Shape'),
            dcc.Dropdown(['Square', 'Circle', 'Line'], 'Square', id='shape'),

        ], style={'width': '24%', 'float': 'right', 'display': 'inline-block'}),
    ], style={
        'padding': '10px 5px'
    }),


    html.Div([
        html.Div([
            dcc.Graph(id='3d_force'),
            dcc.Graph(id='3d_torque')
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
        html.Div([
            dcc.Graph(id='contact_surface_plt'),
            dcc.Graph(id='3d_gamma'),
        ], style={'display': 'inline-block', 'width': '49%'}),
    ], style={
        'padding': '10px 5px'
    }),

    html.Div([
        html.Div([
            html.Label('Linear vel range'),
            dcc.RangeSlider(0, 0.2, id='linear_vel_range',  value=[0, 0.07], allowCross=False),
            html.Label('Angular vel range'),
            dcc.RangeSlider(0, 20, id='angular_vel_range', value=[0, 3], allowCross=False),
            html.Label('gamma'),
            dcc.Slider(0, 0.02, id='gamma', value=0.008),
        ], style={'display': 'inline-block', 'width': '49%'}),

        html.Div([
            html.Label('Linear vel'),
            dcc.Slider(0, 0.2, id='linear_vel_value', value=0.02),
            html.Label('Angular vel'),
            dcc.Slider(0, 5, id='angular_vel_value', value=3),
        ], style={'display': 'inline-block', 'width': '49%'})
    ], style={
        'padding': '10px 5px'
    }),

    html.Div([

        html.Label('Grid shape'),
    dcc.Input(id="grid_shape", value=21, type="number", min=1, step=1),
    html.Label('Grid_size'),
    dcc.Input(id="grid_size", value=1e-3, type="number", min=0),
    html.Label('mu_c'),
    dcc.Input(id="mu_c", value=1, type="number", min=0),
    html.Label('mu_s'),
    dcc.Input(id="mu_s", value=1.5, type="number", min=0),
    html.Label('v_s'),
    dcc.Input(id="v_s", value=1e-2, type="number", min=0),
    html.Label('alpha'),
    dcc.Input(id="alpha", value=2, type="number", min=1, max=2),
    html.Label('s0'),
    dcc.Input(id="s0", value=1e5, type="number", min=0),
    html.Label('s1'),
    dcc.Input(id="s1", value=2e1, type="number", min=0),
    html.Label('s2'),
    dcc.Input(id="s2", value=0.4, type="number", min=0),
    html.Label('resolution'),
    dcc.Input(id="resolution", value=40, type="number", min=0, step=1),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div([
        dcc.Graph(id='ellipsoid'),
    ], style={'display': 'inline-block', 'width': '49%'}),
])


@app.callback(
    [Output('3d_torque', 'figure'),
     Output('3d_torque', 'config')],
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
    Input('gamma', 'value')
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
                    res,
                    gamma):
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
    planar_lugre_reduced = PlanarFrictionReduced(properties=properties)

    properties_ellipse = {'grid_shape': (grid_shape, grid_shape),  # number of grid elements in x any
                          'grid_size': grid_size,  # the physical size of each grid element
                          'mu_c': mu_c,
                          'mu_s': mu_c,
                          'v_s': v_s,
                          'alpha': alpha,
                          's0': s0,
                          's1': s1,
                          's2': 0,
                          'dt': 1e-4}

    planar_lugre_ellipse = PlanarFriction(properties=properties_ellipse)
    f1 = planar_lugre_ellipse.steady_state(vel_vec={'x': 0.1, 'y': 0.1, 'tau': 0},
                                          p_x_y=shape_set[shape])
    f2 = planar_lugre_ellipse.steady_state(vel_vec={'x': 0, 'y': 0, 'tau': 0.1},
                                           p_x_y=shape_set[shape])

    x = np.linspace(lin_vel_range[0], lin_vel_range[1], res)
    y = np.linspace(ang_vel_range[0], ang_vel_range[1], res)
    f_surf = np.zeros((res, res))
    t_surf = np.zeros((res, res))

    f_surf_red = np.zeros((res, res))
    t_surf_red = np.zeros((res, res))

    f_surf_red_no_g = np.zeros((res, res))
    t_surf_red_no_g = np.zeros((res, res))

    f_surf_red_no_g_ellipse = np.zeros((res, res))
    t_surf_red_no_g_ellipse = np.zeros((res, res))


    g_surf = np.zeros((res, res))

    if 'Gamma_max' in model:
        f = planar_lugre_reduced.steady_state_gamma(vel_vec={'x': 0, 'y': 0, 'tau': 1e-9}, p_x_y=shape_set[shape],
                                                    gamma=gamma)
        gamma = planar_lugre_reduced.update_radius({'x': 0, 'y': 0, 'tau': 1e-9})
        print('gamma', gamma)

    ls_approx = CustomHashList(100)

    ls_approx.initialize(planar_lugre_ellipse, shape_set[shape])

    for x_, v in enumerate(x):
        for y_, w in enumerate(y):
            if 'Full' in model:

                f = planar_lugre.steady_state(vel_vec={'x':v, 'y': 0, 'tau': w}, p_x_y=shape_set[shape])
                f_surf[x_, y_] = np.linalg.norm([f['x'], f['y']])
                t_surf[x_, y_] = abs(f['tau'])
            if shape != 'Circle' and 'Reduced + gamma' in model:
                f, g = planar_lugre_reduced.steady_state(vel_vec={'x':v, 'y': 0, 'tau': w}, p_x_y=shape_set[shape])
                f_surf_red[x_, y_] = np.linalg.norm([f['x'], f['y']])
                t_surf_red[x_, y_] = abs(f['tau'])
                g_surf[x_, y_] = g

            if 'Reduced' in model:

                f = planar_lugre_reduced.steady_state_gamma(vel_vec={'x':v, 'y': 0, 'tau': w}, p_x_y=shape_set[shape],
                                                               gamma=gamma)
                f_surf_red_no_g[x_, y_] = np.linalg.norm([f['x'], f['y']])
                t_surf_red_no_g[x_, y_] = abs(f['tau'])

            if 'Ellipse comp' in model:
                #ellipse = np.zeros(2)

                #f = planar_lugre_ellipse.steady_state(vel_vec={'x': v, 'y': 0, 'tau': w},
                #                                      p_x_y=shape_set[shape])
                #ellipse[0] = np.linalg.norm([f['x'], f['y']]) /np.linalg.norm([f1['x'], f1['y']])
                #ellipse[1] = abs(f['tau'])/abs(f2['tau'])
                ellipse = ls_approx.get_interpolation(v, w)
                f = planar_lugre_reduced.steady_state_gamma_ellipse(vel_vec={'x': v, 'y': 0, 'tau': w}, p_x_y=shape_set[shape],
                                                                    gamma=gamma, norm_ellipse=ellipse)
                f_surf_red_no_g_ellipse[x_, y_] = np.linalg.norm([f['x'], f['y']])
                t_surf_red_no_g_ellipse[x_, y_] = abs(f['tau'])


    fig_force = make_subplots()
    fig_torque = go.Figure()

    if 'Full' in model:
        fig_force.add_trace(go.Surface(z=f_surf.T, x=x, y=y))
        fig_torque.add_trace(go.Surface(z=t_surf.T, x=x, y=y))

    if 'Reduced' in model:
        fig_force.add_trace(go.Surface(z=f_surf_red_no_g.T, x=x, y=y, opacity=0.7))
        fig_torque.add_trace(go.Surface(z=t_surf_red_no_g.T, x=x, y=y, opacity=0.7))

    if 'Reduced + gamma' in model:
        fig_force.add_trace(go.Surface(z=f_surf_red.T, x=x, y=y, opacity=0.7))
        fig_torque.add_trace(go.Surface(z=t_surf_red.T, x=x, y=y, opacity=0.7))

    if 'Ellipse comp' in model:
        fig_force.add_trace(go.Surface(z=f_surf_red_no_g_ellipse.T, x=x, y=y, opacity=0.7))
        fig_torque.add_trace(go.Surface(z=t_surf_red_no_g_ellipse.T, x=x, y=y, opacity=0.7))


    fig_force.update_layout(title='Linear force', autosize=False,
                      width=500, height=500,
                      scene=dict(
                          xaxis_title='lin vel',
                          yaxis_title='ang vel',
                          zaxis_title='lin force'),
                      margin=dict(l=0, r=0, b=0, t=30),
                      uirevision=True)
    #fig_torque = go.Figure(data=[go.Surface(z=t_surf.T, x=x, y=y)])
    fig_torque.update_layout(title='Torque', autosize=False,
                      width=500, height=500,
                      scene=dict(
                          xaxis_title='lin vel',
                          yaxis_title='ang vel',
                          zaxis_title='torque'),
                      margin=dict(l=0, r=0, b=0, t=30),
                      uirevision=True)

    fig_gamma = go.Figure(data=[go.Surface(z=g_surf.T, x=x, y=y)])
    fig_gamma.update_layout(title='Gamma radius', autosize=False,
                      width=500, height=500,
                      scene=dict(
                          xaxis_title='lin vel',
                          yaxis_title='ang vel',
                          zaxis_title='radius'),
                      margin=dict(l=0, r=0, b=0, t=30),
                      uirevision=True)
    config = {
        'toImageButtonOptions': {
            'format': 'svg',  # one of png, svg, jpeg, webp
        }
    }
    return [fig_torque, config]


@app.callback(
    [Output('contact_surface_plt', 'figure'),
     Output('contact_surface_plt', 'config')],
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

    config = {
        'toImageButtonOptions': {
            'format': 'svg',  # one of png, svg, jpeg, webp
        }
    }
    return [fig, config]


@app.callback(
    [Output('ellipsoid', 'figure'),
     Output('ellipsoid', 'config')],
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
    """

    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                      xaxis_title="Force",
                      yaxis_title="Torque",
                      margin=dict(l=10, r=10, b=30, t=30),
                      uirevision=True)


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
    config = {
        'toImageButtonOptions': {
            'format': 'svg',  # one of png, svg, jpeg, webp
        }
    }
    return [fig, config]

if __name__ == '__main__':
    app.run_server(debug=True)