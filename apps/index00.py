# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__)

colors = { 'background': '#111111', 'text': '#7FDBFF' }

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

app.layout = html.Div( children=[ html.H1( children='Hello Dash', style={ 'textAlign': 'center','color': colors['text'] } ) ,

            html.Div(children='Dash: A web application framework for your data.', style={'textAlign': 'center', 'color': colors['text'] } )]
            )

if __name__ == '__main__':
    app.run_server(debug=True)
