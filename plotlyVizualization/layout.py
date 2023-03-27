import numpy as np
from dash import html, dcc, Input, Output


def get_layout():
    layout = html.Div([
        html.Div([
            html.Div([
                html.Label('Friction Model'),
                dcc.Checklist(
                    ['Full', 'Reduced with LS', 'Reduced with ellipse'],
                    ['Full'],
                    id='friction_model',
                    labelStyle={'display': 'inline-block', 'marginTop': '5px'}
                )],
                style={'width': '24%', 'display': 'inline-block'}),
            html.Div([
                dcc.Dropdown(id='format_dropdown',
                 options=[
                            {'label': 'JPEG', 'value': 'jpeg'},
                            {'label': 'PNG', 'value': 'png'},
                            {'label': 'SVG', 'value': 'svg'},
                            {'label': 'WebP', 'value': 'webp'}
                         ],
                 value='png'),
            ],
            style={'width': '24%', 'display': 'inline-block'}),

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
                dcc.Dropdown(['Square', 'Circle', 'Line', 'LineGrad'], 'Square', id='shape'),

            ], style={'width': '24%', 'float': 'right', 'display': 'inline-block'}),
        ], style={
            'padding': '10px 5px'
        }),


        html.Div([
            html.Div([
                dcc.Graph(id='3d_force'),
                dcc.Store(id='store_surface_data')
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([
                dcc.Graph(id='3d_torque'),

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
            ], style={'display': 'inline-block', 'width': '49%'}),

            html.Div([
                html.Label('Linear vel'),
                dcc.Slider(0, 0.2, id='linear_vel_value', value=0.02),
                html.Label('Direction'),
                dcc.Slider(0, 2*np.pi, id='direction_value', value=0.0),
                html.Label('Angular vel'),
                dcc.Slider(0, 5, id='angular_vel_value', value=3),
                html.Label('Ratio'),
                dcc.Slider(0, 1, id='ratio', value=0.5),
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
        dcc.Input(id="mu_s", value=1, type="number", min=0),
        html.Label('v_s'),
        dcc.Input(id="v_s", value=1e-2, type="number", min=0),
        html.Label('alpha'),
        dcc.Input(id="alpha", value=2, type="number", min=1, max=2),
        html.Label('s0'),
        dcc.Input(id="s0", value=1e5, type="number", min=0),
        html.Label('s1'),
        dcc.Input(id="s1", value=2e1, type="number", min=0),
        html.Label('s2'),
        dcc.Input(id="s2", value=0, type="number", min=0),
        html.Label('resolution'),
        dcc.Input(id="resolution", value=40, type="number", min=0, step=1),
        ], style={'display': 'inline-block', 'width': '20%'}),

        html.Div([
            dcc.Graph(id='contact_surface_plt'),
        ], style={'display': 'inline-block', 'width': '39%'}),
        html.Div([
            dcc.Graph(id='ellipsoid'),
        ], style={'display': 'inline-block', 'width': '39%'}),

        html.Div([dcc.Graph(id='limit surface'),
                  ], style={'display': 'inline-block', 'width': '85%'}),
    ])

    return layout

