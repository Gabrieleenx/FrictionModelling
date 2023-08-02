from dash import Dash, html, dcc, Input, Output
from plotlyVizualization.utils import *
from frictionModels.utils import vel_to_cop
from plotlyVizualization.layout import get_layout
from surfaces.surfaces import p_square, p_line, p_circle, p_line_grad, PObject
from frictionModels.frictionModel import DistributedFrictionModel, ReducedFrictionModel

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

shape_set = {'Square': p_square, 'Circle': p_circle, 'Line': p_line, 'LineGrad': p_line_grad}

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
    Input('direction_value', 'value'),
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
                        res,
                        direction):

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
    p_obj = PObject(properties['grid_size'], properties['grid_shape'], shape_set[shape])

    if 'Distributed' in model:
        planar_lugre = DistributedFrictionModel(properties=properties)
        planar_lugre.update_p_x_y(p_obj)
        f_surf, t_surf, x_y = calc_surface(planar_lugre, lin_vel_range, ang_vel_range, res, direction)
        data_local = dict()
        data_local['f'] = f_surf
        data_local['tau'] = t_surf
        data_local['x'] = x_y[0]
        data_local['y'] = x_y[1]
        data['Distributed'] = data_local

    if 'Reduced with LS' in model:
        planar_lugre_reduced = ReducedFrictionModel(properties=properties)
        planar_lugre_reduced.update_p_x_y(p_obj)
        planar_lugre_reduced.update_pre_compute()
        f_surf, t_surf, x_y = calc_surface(planar_lugre_reduced, lin_vel_range, ang_vel_range, res, direction)
        data_local = dict()
        data_local['f'] = f_surf
        data_local['tau'] = t_surf
        data_local['x'] = x_y[0]
        data_local['y'] = x_y[1]
        data['Reduced with LS'] = data_local

    if 'Reduced with ellipse' in model:
        planar_lugre_reduced_ellipse = ReducedFrictionModel(properties=properties, ls_active=False)
        planar_lugre_reduced_ellipse.update_p_x_y(p_obj)
        planar_lugre_reduced_ellipse.update_pre_compute()
        f_surf, t_surf, x_y = calc_surface(planar_lugre_reduced_ellipse, lin_vel_range, ang_vel_range, res, direction)
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
    Input('contact_surface', 'value'),
    Input('format_dropdown', 'value'),
    Input('direction_value', 'value'),
    Input('ratio', 'value')
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
                  contact_surface,
                  format_value,
                  direction,
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
    p_obj = PObject(properties['grid_size'], properties['grid_shape'], shape_set[shape])

    planar_lugre = DistributedFrictionModel(properties=properties)
    planar_lugre.update_p_x_y(p_obj)
    cop = planar_lugre.cop
    lin_vel_x = np.cos(direction)*lin_vel
    lin_vel_y = np.sin(direction)*lin_vel


    vx = np.cos(ratio)*lin_vel_x
    vy = np.cos(ratio)*lin_vel_y

    vt =np.sin(ratio)*ang_vel

    vel = vel_to_cop(-cop, vel_vec={'x': vx, 'y': vy, 'tau': vt})


    planar_lugre.step(vel_vec={'x': vel['x'], 'y': vel['y'], 'tau': vel['tau']})
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
    fig.add_trace(go.Scatter(x=[cop[0]], y=[cop[1]]))



    fig.add_trace(go.Scatter(x=[-vel['y']/vel['tau']], y=[vel['x']/vel['tau']]))

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
                    'arrowcolor':'white'
                }
                arrows += [arrow]
        fig.update_layout(annotations=arrows)


    format_value = format_value if format_value is not None else 'png'

    config = {'toImageButtonOptions': {'format': format_value}}

    return [fig, config]


@app.callback(
    [Output('ellipsoid', 'figure'),
     Output('ellipsoid', 'config'),
     Output('limit surface', 'figure')],
    Input('shape', 'value'),
    Input('linear_vel_value', 'value'),
    Input('direction_value', 'value'),
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
    Input('ratio', 'value'),
    Input('format_dropdown', 'value')
)
def update_ellipse(shape,
                   lin_vel,
                   direction,
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
                   ratio,
                   format_value):

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

    p_obj = PObject(properties['grid_size'], properties['grid_shape'], shape_set[shape])

    planar_lugre = DistributedFrictionModel(properties=properties)
    planar_lugre_reduced = ReducedFrictionModel(properties=properties, nr_ls_segments=20)

    planar_lugre.update_p_x_y(p_obj)
    planar_lugre_reduced.update_p_x_y(p_obj)
    planar_lugre_reduced.update_pre_compute()
    planar_lugre_reduced.ls_active = True

    num = 20

    lin_vel_x = np.cos(direction)*lin_vel
    lin_vel_y = np.sin(direction)*lin_vel

    cop = planar_lugre.cop

    fig = go.Figure()
    data = np.zeros((4, 4*num))

    rot_z = np.array([[np.cos(-direction), -np.sin(-direction)],
                      [np.sin(-direction), np.cos(-direction)]])
    ratio_ = np.hstack([np.linspace(0, np.pi/2, num),
                        np.linspace(np.pi/2, np.pi, num),
                        np.linspace(np.pi, 3*np.pi/2, num),
                        np.linspace(3*np.pi/2, 2*np.pi, num)])
    for i in range(4*num):
        vx = np.cos(ratio_[i]) * lin_vel_x
        vy = np.cos(ratio_[i]) * lin_vel_y
        vt = np.sin(ratio_[i]) * ang_vel
        f = planar_lugre.step(vel_vec=vel_to_cop(-cop, vel_vec={'x': vx, 'y': vy, 'tau': vt}))
        f = planar_lugre.force_at_cop
        data[0, i] = f['x']
        data[1, i] = f['y']
        data[2, i] = f['tau']
        data[3, i] = rot_z.dot(np.array([f['x'], f['y']]).T)[0]

    max_save = np.max(data, axis=1)
    max_save[max_save==0] = 1
    data = data.T / max_save
    data = data.T

    for i in range(4):
        if i % 2 == 0:
            name_ = str('Limit surface full')
            dash_ = None
            color_ = 'midnightblue'
        else:
            name_ = str('Limit surface full mirror CoR')
            dash_ = "dash"
            color_ = "skyblue"
        fig.add_trace(go.Scatter(x=data[3, i*num:(i+1)*num], y=data[2, i*num:(i+1)*num],
                                 line=go.scatter.Line(color=color_, dash=dash_), name=name_))

    data_red = np.zeros((4, 4*num))
    for i in range(4*num):
        vx = np.cos(ratio_[i]) * lin_vel_x
        vy = np.cos(ratio_[i]) * lin_vel_y
        vt = np.sin(ratio_[i]) * ang_vel
        f = planar_lugre_reduced.step(vel_vec=vel_to_cop(-cop, vel_vec={'x': vx, 'y': vy, 'tau': vt}))
        #f = planar_lugre_reduced.step(vel_vec={'x': lin_vel_x_list[i], 'y': lin_vel_y_list[i], 'tau': ang_vel_list[i]})
        f = planar_lugre_reduced.force_at_cop

        data_red[0, i] = f['x']
        data_red[1, i] = f['y']
        data_red[2, i] = f['tau']
        data_red[3, i] = rot_z.dot(np.array([f['x'], f['y']]).T)[0]


    data_red = data_red.T / max_save
    data_red = data_red.T

    for i in range(4):
        if i % 2 == 0:
            name_ = str('Limit surface red')
            dash_ = None
            color_ = 'firebrick'
        else:
            name_ = str('Limit surface red mirror CoR')
            dash_ = "dash"
            color_ = "indianred"
        fig.add_trace(go.Scatter(x=data_red[3, i*num:(i+1)*num], y=data_red[2, i*num:(i+1)*num],
                                 line=go.scatter.Line(color=color_, dash=dash_), name=name_))


    arrows = []
    vx = np.cos(ratio)*lin_vel_x
    vy = np.cos(ratio)*lin_vel_y
    vt = np.sin(ratio)*ang_vel
    v3 = vel_to_cop(-cop, vel_vec={'x': vx, 'y': vy, 'tau': vt})

    f = planar_lugre.step(v3)
    f = planar_lugre.force_at_cop

    dx = rot_z.dot(np.array([f['x'], f['y']]).T)[0]/ max_save[3]
    dy = f['tau'] / max_save[2]

    arrows += [return_arrow(0, 0, dx, dy, 'blue')]
    f = planar_lugre_reduced.step(v3)
    f = planar_lugre_reduced.force_at_cop

    dx = rot_z.dot(np.array([f['x'], f['y']]).T)[0] / max_save[3]
    dy = f['tau'] / max_save[2]

    arrows += [return_arrow(0, 0, dx, dy, 'green')]

    fig.update_layout(annotations=arrows)
    fig.update_layout(title='Ellipse',
                      width=600, height=500,
                      xaxis_title='Force',
                      yaxis_title='Torque',
                      yaxis=dict(scaleanchor="x", scaleratio=1)
                      )

    format_value = format_value if format_value is not None else 'png'

    config = {'toImageButtonOptions': {'format': format_value}}

    data3d = np.zeros((3, 4*num*(2*num)))
    directione3 = np.linspace(0, 2*np.pi, 4*num)
    for j in range(4*num):
        direction = directione3[j]
        lin_vel_x = np.cos(direction) * lin_vel
        lin_vel_y = np.sin(direction) * lin_vel

        cop = planar_lugre.cop
        v1 = vel_to_cop(-cop, vel_vec={'x': 0, 'y': 0, 'tau': ang_vel})
        v2 = vel_to_cop(-cop, vel_vec={'x': lin_vel_x, 'y': lin_vel_y, 'tau': 0})

        ang_vel_list = np.linspace(v1['tau'], v2['tau'], num)
        lin_vel_x_list = np.linspace(v1['x'], v2['x'], num)
        lin_vel_y_list = np.linspace(v1['y'], v2['y'], num)

        for i in range(num):
            f = planar_lugre_reduced.step(vel_vec={'x': lin_vel_x_list[i], 'y': lin_vel_y_list[i], 'tau': ang_vel_list[i]})
            f = planar_lugre_reduced.force_at_cop
            data3d[0, i+j*num] = f['x']
            data3d[1, i+j*num] = f['y']
            data3d[2, i+j*num] = f['tau']
            data3d[0, i + j * num + 4*num*num] = -f['x']
            data3d[1, i + j * num + 4*num*num] = -f['y']
            data3d[2, i + j * num +4*num*num] = -f['tau']
    max_save3d = np.max(abs(data3d), axis=1)
    data3d = data3d.T / max_save3d
    data3d = data3d.T
    fig_2 = go.Figure()
    fig_2.add_trace(go.Scatter3d(x=data3d[0,:], y=data3d[1,:], z=data3d[2,:],
                                 mode='markers',
                                 marker=dict(
                                     size=8,
                                     color=data3d[2,:],  # set color to an array/list of desired values
                                     colorscale='Viridis',  # choose a colorscale
                                     opacity=1
                                 )
                                 ))

    fig_2.update_layout(title='Limit surface',
                      width=1000, height=1000,
                      )
    return [fig, config, fig_2]


if __name__ == '__main__':
    app.run_server(debug=True)