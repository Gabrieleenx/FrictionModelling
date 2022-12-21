import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
from plotly import graph_objects as go
from plotly.subplots import make_subplots

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div("Please select the saving format below:", id="my_div"),
    dcc.Dropdown(id='format_dropdown',
                 options=[
                            {'label': 'JPEG', 'value': 'jpeg'},
                            {'label': 'PNG', 'value': 'png'},
                            {'label': 'SVG', 'value': 'svg'},
                            {'label': 'WebP', 'value': 'webp'}
                         ],
                 value='jpeg'),
    dcc.Graph(id='my_graph'),

])


@app.callback(
    [Output('my_graph', 'figure'),
     Output('my_graph', 'config')],
    [Input('format_dropdown', 'value')]
)
def update_graph(format_value):

    format_value = format_value if format_value is not None else 'png'
    print(format_value)

    fig = make_subplots()
    fig.update_layout(title="A graph")
    fig.update_xaxes(title_text="Time (s)", rangemode="nonnegative")
    fig.update_yaxes(title_text="Some value (unit)", rangemode="nonnegative")

    fig.add_trace(go.Scatter(x=[1, 2, 3],
                             y=[1, 2, 3],
                             name='Some data',
                             mode='lines+markers',
                             marker=dict(color='blue')))

    config = {
        'toImageButtonOptions': {
            'format': format_value,  # one of png, svg, jpeg, webp
        }
    }

    return [fig, config]


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=5000)