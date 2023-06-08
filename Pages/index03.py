import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from collections import OrderedDict
import dash_daq as daq

import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score,accuracy_score

import pandas as pd

df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")
df = pd.DataFrame(
    OrderedDict([(name, col_data) for (name, col_data) in df.items()])
)

dash.register_page("Test",  path='/Testing',

layout = html.Div([html.Div(children=[

    dcc.Store(id='store-target2', storage_type='local'),
    dcc.Store(id='store-split2', storage_type='local'),

html.P("Select Model ( For Testing )", className="control_label"),
    dcc.Dropdown(
        id="select_test",
        options=['LogistcRegression', 'RandomForestClassifier', 'ExtraTreesClassifier','SGDClassifier'],
        multi=False,
        value=None,
        clearable=True,        
),
html.Hr(),
html.P("Select Target (Y column)", className="control_label"),
    dcc.Dropdown(
        id="select_target2",
        options=[{'label':x, 'value':x} for x in df.columns],
        multi=False,
        value=None,
        clearable=True       
),
html.Div(id='output-target'),
html.Hr(),
daq.Slider(id='slider2',
    min=0,
    max=100,
    value=100,
    handleLabel={"showCurrentValue": True,"label": "VALUE"},
    step=10,

),
html.Div(id='output-slider'),
html.Hr(),
html.Div(children='Please select taget model and training split first',id='select-test-output2'),
html.Div([
    daq.LEDDisplay(
        id='precision',
        label="Precision",
        value=0,
        style={
                        'height':'100px',
                        'margin-left':'10px',
                        'text-align':'center',
                        'width':'30%',
                        'display':'inline-block'
               }
    ),daq.LEDDisplay(
        id='recall',
        label="Recall",
        value=0,
        style={
                        'height':'100px',
                        'margin-left':'10px',
                        'text-align':'center',
                        'width':'30%',
                        'display':'inline-block'
               }
    ),daq.LEDDisplay(
        id='accuracy',
        label="Accuracy",
        value=0,
        style={
                        'height':'100px',
                        'margin-left':'10px',
                        'text-align':'center',
                        'width':'30%',
                        'display':'inline-block'
               }
    ),

]),

dcc.Store(id='store-score', storage_type='local'),
html.Hr(),
html.H3(children='Multiclass ROC Curve'),
dcc.Graph(id="roc-grph"),

])
])
)

@callback(Output('output-target', 'children'),
          Input('select_target2', 'value'))
def clean_data(cc):
    if cc is not None:
        return f'You have select : {str(cc)} to be target column'
    else :
        raise PreventUpdate
    
@callback(Output('store-target2', 'value'),
          Input('select_target2', 'value'))
def clean_data(cc):
    if cc is not None:
        return str(cc)
    else :
        raise PreventUpdate

@callback(
    Output('select-test-output', 'children'),
    Input('select_test', 'value')
)
def update_output(value):
    if value is not None :
        return f'You have selected : {value} Model'
    else :
        raise PreventUpdate

@callback(
    Output('output-slider', 'children'),
    Input('slider2', 'value')
)
def update_output(value):
    if value is not None :
        return f'You have split training set : {str(value)} %'
    else :
        raise PreventUpdate
    
@callback(
    Output('store-split2', 'value'),
    Input('slider2', 'value')
)
def update_output(value):
    if value is not None :
        return str(value)
    else :
        raise PreventUpdate
    
##################################################
###############      score       #################
##################################################
    
@callback(
    Output('precision', 'value'),
    Input('store-score', 'data')
)
def update_output(value):
    if value is not None :
        value = value['precision']
        return value
    else :
        raise PreventUpdate
    
@callback(
    Output('recall', 'value'),
    Input('store-score', 'data')
)
def update_output(value):
    if value is not None :
        value = value['recall']
        return value
    else :
        raise PreventUpdate
    
@callback(
    Output('accuracy', 'value'),
    Input('store-score', 'data')
)
def update_output(value):
    if value is not None :
        value = value['accuracy']
        return value
    else :
        raise PreventUpdate

##################################################

@callback(Output('store-score', 'data'),
              Input('store-target2', 'value'),
              Input('store-split2', 'value'),
              Input('select_test', 'value')
              )
def update_output(targ,value,model):
    if ( targ is not None ) and ( value != '100' ) and (model is not None) :
        df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")
        
        x = df.drop(columns=targ)
        y = df[targ]
        tts = 1-(int(value)/100)
        X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = tts, random_state = 42, stratify = y )

        scora = {}

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
            sc = round(precision_score(y_test, y_pred, average='macro'),2)
            sc1 = round(recall_score(y_test, y_pred, average='macro'),2)
            sc2 = round(accuracy_score(y_test, y_pred)*100,1)

            scora['precision'] = sc
            scora['recall'] = sc1
            scora['accuracy'] = sc2
            
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
            sc = round(precision_score(y_test, y_pred, average='macro'),2)
            sc1 = round(recall_score(y_test, y_pred, average='macro'),2)
            sc2 = round(accuracy_score(y_test, y_pred)*100,1)

            scora['precision'] = sc
            scora['recall'] = sc1
            scora['accuracy'] = sc2

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
            sc = round(precision_score(y_test, y_pred, average='macro'),2)
            sc1 = round(recall_score(y_test, y_pred, average='macro'),2)
            sc2 = round(accuracy_score(y_test, y_pred)*100,1)

            scora['precision'] = sc
            scora['recall'] = sc1
            scora['accuracy'] = sc2

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
            sc = round(precision_score(y_test, y_pred, average='macro'),2)
            sc1 = round(recall_score(y_test, y_pred, average='macro'),2)
            sc2 = round(accuracy_score(y_test, y_pred)*100,1)

            scora['precision'] = sc
            scora['recall'] = sc1
            scora['accuracy'] = sc2

        else :
            raise PreventUpdate
        
        return scora

##################################################
###################   Graph   ####################
##################################################

@callback(Output('roc-grph', 'figure'),
              Input('store-target2', 'value'),
              Input('store-split2', 'value'),
              Input('select_test', 'value')
              )
def update_output(targ,value,model):
    if ( targ is not None ) and ( value != '100' ) and (model is not None) :
        df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")
        
        x = df.drop(columns=targ)
        y = df[targ]

        tts = 1-(int(value)/100)
        X , y = train_test_split( x, y, test_size = tts, random_state = 42, stratify = y )

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
            pipeline.fit(X,y)
            y_pred = pipeline.predict_proba(X)

            y_onehot = pd.get_dummies(y, columns=pipeline.classes_)
    
            # Create an empty figure, and iteratively add new lines
            # every time we compute a new class
            fig = go.Figure()
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1)
            for i in range(y_pred.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_pred[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500
            )
            return fig
            
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
            pipeline.fit(X,y)
            y_pred = pipeline.predict_proba(X)

            y_onehot = pd.get_dummies(y, columns=pipeline.classes_)
    
            # Create an empty figure, and iteratively add new lines
            # every time we compute a new class
            fig = go.Figure()
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1)
            for i in range(y_pred.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_pred[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500
            )
            return fig

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
            pipeline.fit(X,y)
            y_pred = pipeline.predict_proba(X)

            y_onehot = pd.get_dummies(y, columns=pipeline.classes_)
    
            # Create an empty figure, and iteratively add new lines
            # every time we compute a new class
            fig = go.Figure()
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1)
            for i in range(y_pred.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_pred[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500
            )
            return fig

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
            pipeline.fit(X,y)
            y_pred = pipeline.predict_proba(X)

            y_onehot = pd.get_dummies(y, columns=pipeline.classes_)
    
            # Create an empty figure, and iteratively add new lines
            # every time we compute a new class
            fig = go.Figure()
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1)
            for i in range(y_pred.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_pred[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500
            )
            return fig
        
        else :
            raise PreventUpdate
        