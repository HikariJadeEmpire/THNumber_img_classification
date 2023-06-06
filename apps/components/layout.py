from dash import Dash,html,dcc
from components import upload_input

def create_layout( app : Dash ):
    return html.Div( className="app-div",
                    children=[html.H1(app.title),html.Hr(),
                              html.Div(className='upload',children=[
                                  upload_input.input(app)
                              ])]
                                                 )