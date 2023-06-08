import dash
from dash.dependencies import Input, Output
from dash import dcc, html, callback
from dash import dash_table
from collections import OrderedDict
from dash.exceptions import PreventUpdate
import plotly.express as px

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score,accuracy_score

import dash_daq as daq
import pandas as pd

df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")
df = pd.DataFrame(
    OrderedDict([(name, col_data) for (name, col_data) in df.items()])
)

dash.register_page("Train",  path='/Training',

layout = html.Div([ html.Div(children=[
            html.H4(children='Select your training'),
            html.Div(children='select Y column & choose trainning number'),
            html.Hr(),

            html.Div([
        dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],page_size=10, id='tbl')
    ]),

html.Hr(),

    html.P("Select Target (Y column)", className="control_label"),
    dcc.Dropdown(
        id="select_target",
        options=[{'label':x, 'value':x} for x in df.columns],
        multi=False,
        value=None,
        clearable=False,
        persistence="local"        
),
html.Div(id='dd-output-container'),
dcc.Store(id='store-target', storage_type='local'),

html.Hr(),
html.Div(children='select your training number'),

html.Div([daq.LEDDisplay(
        id='my-LED-display-1',
        label="Trainning splits",
        value=100,
    ),
    dcc.Slider(
        id='my-LED-display-slider-1',
        min=0,
        max=100,
        step=10,
        value=100
    )])
]), dcc.Store(id='store-split', storage_type='local'),
html.Hr(),

html.Div(children='select tools for Cross Validation'),
html.Div(dcc.Dropdown(
    id="selectcv",
    options = ['LogistcRegression', 'RandomForestClassifier', 'ExtraTreesClassifier','SGDClassifier'],
    multi = True ,
    value = None ,
    clearable = False,
    persistence="local"

), style={'padding': 10, 'flex': 1}), 
html.Div(children='Please select the models',id='cvscore'),
dcc.Store(id='output-cv', storage_type='local'),
dcc.Store(id='output-cv2', storage_type='local'),
html.Hr(),

 html.Div(children=[
            html.H3(children='Comparing the Accuracy Scores of Different Models (Bar Plot)'),

            html.Div(children='Select a model'),

            dcc.Graph(
                id='id2',
                figure = {},
            )
        ], style={'padding': 10, 'flex': 1}),

])

)

@callback(
    Output('cvscore', 'children'),
    Input('output-cv', 'value'),
    Input('output-cv2', 'children')
)
def update_output(data,children):
    if (data is not None) and (children != []) :
        scorr = {}
        for i in data:
            for j in children:
                if i==j:
                    scorr[i]=data[i]
        return f'Accuracy score : {scorr}'
    else :
        raise PreventUpdate

@callback(Output('id2', 'figure'),
          Input('output-cv', 'value'),
          Input('output-cv2', 'children')
)
def upd_fig(cc,select):
    if (cc is not None) and (select != []):
        scor = {}
        for i in cc:
            for j in select:
                if i==j:
                    scor[i]=cc[i]
        cc = {'model':scor.keys(),'acc_score':scor.values()}
        cc = pd.DataFrame(cc)
        figure = px.bar(cc,x='acc_score',y='model',
             color='acc_score',
             labels={'acc_score':'Accuracy score'}, height=400)
        figure.update_layout(transition_duration=1)
        return figure
    else :
        raise PreventUpdate

@callback(Output('store-target', 'value'),
          Input('select_target', 'value'))
def clean_data(cc):
    if cc is not None:
        return cc
    else :
        raise PreventUpdate

@callback(
    Output('my-LED-display-1', 'value'),
    Input('my-LED-display-slider-1', 'value')
)
def update_output(value):
    return str(value)

@callback(
    Output('store-split', 'value'),
    Input('my-LED-display-slider-1', 'value')
)
def update_led(value):
    return str(value)


@callback(
    Output('output-cv2', 'children'),
    Input('selectcv', 'value')
)
def update_output(value):
    if value is not None :
        return value


@callback(
    Output('dd-output-container', 'children'),
    Input('select_target', 'value')
)
def update_output(value):
    return f'You have select : {value}'

@callback(Output('output-cv', 'value'),
              Input('store-target', 'value'),
              Input('store-split', 'value')
              )
def update_output(cc,value):
    if ( cc is not None ) and ( value != '100' ) :
        df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")

        x = df.drop(columns=cc)
        y = df[cc]
        tts = 1-(int(value)/100)
        X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = tts, random_state = 42, stratify = y )

        score = {}
        for i in ['LogistcRegression', 'RandomForestClassifier', 'ExtraTreesClassifier','SGDClassifier']:
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
                pr = pipeline.fit(X_train, y_train)
                y_pred = pr.predict(X_test)
                sc = round(accuracy_score(y_test, y_pred)*100,1)

                score['LogistcRegression']=sc
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
                pr = pipeline.fit(X_train, y_train)
                y_pred = pr.predict(X_test)
                sc = round(accuracy_score(y_test, y_pred)*100,1)

                score['RandomForestClassifier']=sc
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
                pr = pipeline.fit(X_train, y_train)
                y_pred = pr.predict(X_test)
                sc = round(accuracy_score(y_test, y_pred)*100,1)

                score['ExtraTreesClassifier']=sc
            elif i == 'SGDClassifier' :
                steps = [
                ('scalar', MinMaxScaler()),
                ('SGDClassifier',SGDClassifier(alpha=0.0001, average=False, class_weight=None,
               early_stopping=False, epsilon=0.1, eta0=0.001, fit_intercept=True,
               l1_ratio=0.15, learning_rate='optimal', loss='hinge',
               max_iter=1000, n_iter_no_change=5, n_jobs=-1, penalty='l2',
               power_t=0.5, random_state=123, shuffle=True, tol=0.001,
               validation_fraction=0.1, verbose=0, warm_start=False))
                            ]
                pipeline = Pipeline(steps)
                pr = pipeline.fit(X_train, y_train)
                y_pred = pr.predict(X_test)
                sc = round(accuracy_score(y_test, y_pred)*100,1)

                score['SGDClassifier']=sc
        return score

