import dash
from dash.dependencies import Input, Output
from dash import dcc, html, callback
from dash import dash_table
from collections import OrderedDict

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report 

import dash_daq as daq
import pandas as pd

df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")
ycol = None
df = pd.DataFrame(
    OrderedDict([(name, col_data) for (name, col_data) in df.items()])
)

dash.register_page("Train",  path='/Training',

layout = html.Div([html.Hr(),
                   html.Div(children=[
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
        value=ycol,
        clearable=False,        
),
html.Div(id='dd-output-container'),
dcc.Store(id='store-target'),
html.Hr(),
html.Div(children='select your training number'),

html.Div([daq.LEDDisplay(
        id='my-LED-display-1',
        label="Trainning splits",
        value=100
    ),
    dcc.Slider(
        id='my-LED-display-slider-1',
        min=0,
        max=100,
        step=10,
        value=100
    )])
]), dcc.Store(id='store-split'),
html.Hr(),

html.Div(children='select tools for Cross Validation'),
dcc.Dropdown(
    id="selectcv",
    options = ['LogistcRegression', 'RandomForestClassifier', 'ExtraTreesClassifier','SGDClassifier'],
    multi = True ,
    value = None ,
    clearable = True

), 
html.Div(children='Cross validation score',id='cvscore'),
dcc.Store(id='output-cv'),
dcc.Store(id='output-cv2'),
html.Hr(),

])

)


@callback(Output('store-target', 'value'),
          Input('select_target', 'value'))
def clean_data(cc):
    if cc is not None:
        return cc
    else :
        return None


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
    Output('cvscore', 'children'),
    Input('output-cv', 'data'),
    Input('output-cv2', 'children')
)
def update_output(data,children):
    if (data is not None) and (children != []) :
        scor = {}
        for i in data:
            for j in children:
                if i==j:
                    scor[i]=data[i]/1000
        return f'CV training score : {scor}'
    else :
        return f'Please select your training model'

@callback(
    Output('dd-output-container', 'children'),
    Input('select_target', 'value')
)
def update_output(value):
    return f'You have select : {value}'

@callback(Output('output-cv', 'data'),
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
                sc = pipeline.score(X_train, y_train)

                score['LogistcRegression']=sc*1000
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
                sc = pipeline.score(X_train, y_train)

                score['RandomForestClassifier']=sc*1000
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
                pr = pipeline.fit(X_train, y_train)
                sc = pipeline.score(X_train, y_train)

                score['ExtraTreesClassifier']=sc*1000
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
                sc = pipeline.score(X_train, y_train)

                score['SGDClassifier']=sc*1000
        return score

