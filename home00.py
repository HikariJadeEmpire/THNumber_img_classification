from dash import Dash, html, dcc, callback
import dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True, use_pages=True)

app.layout = html.Div([
	html.H1('Welcome to ML classification APP.'),

    html.Div(
        [ html.Div(dcc.Link(
                    f"{page['name']} - {page['path']}", href=page["relative_path"])
            )
            for page in dash.page_registry.values()
        ]
    ),

	dash.page_container,
	 html.Div(children=[
            html.Div(children='Start your journey with Home page'),
     ])
])

if __name__ == '__main__':
	app.run_server(debug=True)