import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, callback
from dash.exceptions import PreventUpdate

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 

import dash_daq as daq
import pandas as pd

df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")
ycol = None


dash.register_page("Train",  path='/Training',

layout = html.Div([html.Hr(),
                   html.Div(children=[
            html.H4(children='Select your training'),
            html.Div(children='select Y column & choose trainning number'),
            html.Hr(),

    html.P("Select Target", className="control_label"),
    dcc.Dropdown(
        id="select_target",
        options=[{'label':x, 'value':x} for x in df.columns],
        multi=False,
        value=ycol,
        clearable=False,        
),
html.Div(id='dd-output-container'),
dcc.Store(id='store-value'),
html.Hr(),
html.Div(children='select your training number'),

html.Div([daq.LEDDisplay(
        id='my-LED-display-1',
        label="splits",
        value=1
    ),
    dcc.Slider(
        id='my-LED-display-slider-1',
        min=0,
        max=1,
        step=0.05,
        value=1
    )])
]), 
dcc.Store(id='store-value2'),
html.Hr(),
html.Div(children='select tools for Cross Validation'),
dcc.Dropdown(
    id="selectcv",
    options = ['LogistcRegression', 'RandomForestClassifier', 'ExtraTreesClassifier'],
    multi = True ,
    value = None ,
    clearable = True

), dcc.Store(id='output-cv'),

])
)

@callback(Output('store-value', 'data'),
          Input('select_target', 'value'))
def clean_data(value):
     # some expensive data processing step
    if value is not None:
        x = df.drop(columns= str(value)).to_json(date_format='iso', orient='split')
        y = df[str(value)].to_json(date_format='iso', orient='split')
        return x,y
    else :
        return None

@callback(
    Output('my-LED-display-1', 'value'),
    Input('my-LED-display-slider-1', 'value')
)
def update_output(value):
    return str(value)

@callback(
    Output('store-value2', 'value'),
    Input('my-LED-display-slider-1', 'value')
)
def update_output(value):
    cc = value
    return cc


@callback(
    Output('dd-output-container', 'children'),
    Input('select_target', 'value')
)
def update_output(value):
    return f'You have select : {value}'
             

@callback(
    Output('output-cv', 'children'),
    Input('store-value', 'data')
)
#def update_output(value):
    #return f'You have select : {value}'

def update_scv(value):
    if value is not None :
        x = pd.read_json(x, orient='split')
        y = pd.read_json(y, orient='split')
        X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = (1-cc), random_state = 42, stratify = y )

        score = []
        for i in value:
            if i == 'LogistcRegression' :
                steps = [
                ('scalar', MinMaxScaler()),
                ('LogisticRegression',LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, l1_ratio=None, max_iter=1000,
                            multi_class='auto', n_jobs=None, penalty='l2',
                            random_state=123, solver='lbfgs', tol=0.0001, verbose=0,
                            warm_start=False))
                            ]
                pipeline = Pipeline(steps)
                sc = pipeline.score(X_train, y_train)

                score.append(f'LogistcRegression training score : {sc}')
            elif i == 'RandomForestClassifier' :
                steps = [
                ('scalar', MinMaxScaler()),
                ('Randomforest',RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                            criterion='gini', max_depth=None, max_features='sqrt',
                            max_leaf_nodes=None, max_samples=None,
                            min_impurity_decrease=0.0, min_samples_leaf=1,
                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                            n_estimators=100, n_jobs=-1, oob_score=False,
                            random_state=123, verbose=0, warm_start=False))
                            ]
                pipeline = Pipeline(steps)

                sc = pipeline.score(X_train, y_train)

                score.append(f'Randomforest training score : {sc}')
            elif i == 'ExtraTreesClassifier' :
                steps = [
                ('scalar', MinMaxScaler()),
                ('ExtraTreesClassifier',ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                        criterion='gini', max_depth=None, max_features='sqrt',
                        max_leaf_nodes=None, max_samples=None,
                        min_impurity_decrease=0.0, min_samples_leaf=1,
                        min_samples_split=2, min_weight_fraction_leaf=0.0,
                        n_estimators=100, n_jobs=-1, oob_score=False,
                        random_state=123, verbose=0, warm_start=False))
                            ]
                pipeline = Pipeline(steps)

                sc = pipeline.score(X_train, y_train)

                score.append(f'ExtraTreesClassifier training score : {sc}')
        return score
    else :
        return None
