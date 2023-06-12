from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(
    __name__,
    use_pages=True,
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.layout = html.Div([
	html.H1('Welcome to ML classification APP.'),
    html.Div(
        [ html.Div(dcc.Link(
                    f"{page['name']} - {page['path']}", href=page["relative_path"])
            )
            for page in dash.page_registry.values()
        ]
    ),
    html.Hr(),
	dash.page_container,html.Hr(),
	html.Div(children=[
            html.Div(children='Start your journey with Home page'),

     ])
])

if __name__ == '__main__':
	app.run_server(debug=True)