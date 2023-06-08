import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

import pandas as pd

dash.register_page("Test",  path='/Testing',

layout = html.Div([html.Hr(),html.Div(children=[      

dcc.Store(id='output-cv3'),

html.P("Select Model ( For Testing )", className="control_label"),
    dcc.Dropdown(
        id="select_test",
        options=['LogistcRegression', 'RandomForestClassifier', 'ExtraTreesClassifier','SGDClassifier'],
        multi=False,
        value=None,
        clearable=True,        
),
html.Div(id='select-test-output'),

])
])
)

@callback(
    Output('select-test-output', 'children'),
    Input('select_test', 'value')
)
def update_output(value):
    if value is not None :
        return f'You have selected : {str(value)} Model'

@callback(Output('output-cv3', 'data'),
              Input('store-target', 'value'),
              Input('store-split', 'value'),
              Input('select_test', 'value')
              )
def update_output(cc,value,model):
    if ( cc is not None ) and ( value != '100' ) and (model is not None) :
        df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")

        x = df.drop(columns=cc)
        y = df[cc]
        tts = 1-(int(value)/100)
        X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = tts, random_state = 42, stratify = y )

        if model == 'LogistcRegression':
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
            
        elif model == 'RandomForestClassifier' :
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

        elif model == 'ExtraTreesClassifier' :
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

        elif model == 'SGDClassifier' :
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
