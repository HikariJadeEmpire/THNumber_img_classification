from dash import Dash,html,dcc

def input( app : Dash ):
    return html.Div(
        id='upload-image',
        children=[html.H6("Upload"),
                  dcc.Upload( ['Drag and Drop or ', html.A('Select a File')], 
        style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center'
    },
    multiple=True ),
    html.Div(id='output-image-upload')
    ])
                                