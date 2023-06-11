# import dash
# from dash import dcc, html, callback
# from dash.dependencies import Input, Output
# import dash_daq as daq
# from dash.exceptions import PreventUpdate

# import plotly.graph_objects as go
# import plotly.express as px
# import numpy as np

# dash.register_page("Predict",  path='/Predict',

# layout = html.Div([html.Div(children=[

# html.Div(children=[
#             html.H4(children='Import your DATA (for prediction (e.g.\'img.png\'))'),

#             html.Div(children='your DATA must be .png file (if not, please make it)'),
#             html.Hr(),
#     dcc.Upload(
#         id='upload-img',
#         children=html.Div([
#             'Drag and Drop or ',
#             html.A('Select Files')
#         ]),
#         style={
#             'width': '100%',
#             'height': '60px',
#             'lineHeight': '60px',
#             'borderWidth': '1px',
#             'borderStyle': 'dashed',
#             'borderRadius': '5px',
#             'textAlign': 'center',
#             'margin': '10px'
#         },
#         # Allow multiple files to be uploaded
#         multiple=True
#     ),
#     html.Ul(id='output-img')
# ]),
# ])
# ])
# )

# ################################################

# @callback(Output('', ''),
#             Input('upload-data', 'filename'),
#             Input('upload-data', 'contents'))
# def clean_data(x):
#     if x is not None:
#         return f'You have select : {str(x)} to be target column'
#     else :
#         raise PreventUpdate