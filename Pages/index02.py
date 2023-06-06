import dash
from dash.dependencies import Input, Output
from dash import dcc, html, callback
import dash_daq as daq

dash.register_page("Train",  path='/Training',

layout = html.Div([html.Hr(),
                   html.Div(children=[
            html.H4(children='Select your training'),
            html.Div(children='your DATA must be .csv file (if not, please make it)'),
            html.Hr(),
        daq.Slider(id = 'slider',
        min=0,
        max=100,
        value=70,
        handleLabel={"showCurrentValue": True,"label": "SPLIT"},
        step=10
    ),
    html.Div(id='slider-output-1')
])
])
)

@callback(
    Output('slider-output-1', 'children'),
    Input('my-daq-slider-ex-1', 'value')
)
def update_output(value):
    return f'The slider is currently at {value}.'
