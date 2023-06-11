import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import dash_daq as daq
from dash.exceptions import PreventUpdate
import datetime

import plotly.graph_objects as go
import plotly.express as px
import numpy as np

dash.register_page("Predict",  path='/Predict',

layout = html.Div([
    html.Div(children='Please upload image (.png) to predict',id='text0'),
    dcc.Upload(
        id='upload-img',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    
])
)

################################################

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

################################################

@callback(Output('output-image-upload', 'children'),
              Input('upload-img', 'contents'),
              State('upload-img', 'filename'),
              State('upload-img', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children