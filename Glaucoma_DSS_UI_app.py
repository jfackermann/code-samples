# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go

# load patient data
df = pd.read_csv('~/Dropbox/PycharmProjects/Glaucoma_Analyses/Glaucoma_Data_Dec16_Plus.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# params for main dropdown menu
dropdown_opts = [dict([('label', str(count+1)), ('value', str(count))]) for count in range(10)]

# models
models = dict()
models['cdr_avgt | psd'] = {'features': ['cdr_avgt', 'psdonl'],
                            'labels': ['CDR_AVGT', 'PSD'],
                            'scale_x': np.arange(-3, 3, .05),
                            'scale_y': np.arange(-.5, 1.5, .05),
                            'xrange': [-3, 3],
                            'yrange': [-1, 2],
                            'xtickvals': [-2, 0, 2],
                            'ytickvals': [0, 1]}
models['cdr_avgt | ght'] = {'features': ['cdr_avgt', 'ght'],
                            'labels': ['CDR_AVGT', 'GHT'],
                            'scale_x': np.arange(-3, 3, .05),
                            'scale_y': np.arange(.5, 3.5, .05),
                            'xrange': [-3, 3],
                            'yrange': [0, 4],
                            'xtickvals': [-2, 0, 2],
                            'ytickvals': [1, 2, 3]}
models['cdr_avgt | ght_psd'] = {'features': ['cdr_avgt', 'ght_psd'],
                                'labels': ['CDR_AVGT', 'GHT_PSD'],
                                'scale_x': np.arange(-3, 3, .05),
                                'scale_y': np.arange(-3, 3, .05),
                                'xrange': [-3, 3],
                                'yrange': [-3, 3],
                                'xtickvals': [-2, 0, 2],
                                'ytickvals': [-2, 0, 2]}
models['cdr | psd'] = {'features': ['cdratio', 'psdonl'],
                        'labels': ['CDR', 'PSD'],
                        'scale_x': np.arange(-.25, 1.25, .01),
                        'scale_y': np.arange(-.5, 1.5, .01),
                        'xrange': [-.5, 1.5],
                        'yrange': [-.5, 1.5],
                        'xtickvals': [0, .5, 1],
                        'ytickvals': [0, 1]}

# header style
left_margin = '20px'
header_style = {'fontFamily': 'Avenir Next', 'color': '#869bb7', 'marginLeft': left_margin}

# get probability of a positive diagnosis
def get_prob(probs, p1, p2):
    return (np.product(probs) * p1) /     \
            ((np.product(probs) * p1) +   \
            (np.product(1 - probs) * p2))

# get model decision bounds
def decision_bounds(scale_x, scale_y, features, target, model, df):
    # extract features and targets
    X = df[features].values
    Y = df[target].values
    # fit model to training data
    model.fit(X, Y)
    # get model predictions for all values
    xx, yy = np.meshgrid(scale_x, scale_y)
    XX = np.c_[xx.ravel(), yy.ravel()]
    pred = model.predict(XX)
    return pred

# model plot callback function
def do_model_plots(model_index):
    def callback(patient_index):
        if patient_index == None:
           df0 = df
           neg_color = 'rgba(255, 255, 255, 0)'
           pos_color = 'rgba(255, 255, 255, 0)'
        else:
            patient_index = int(patient_index)
            df0 = df.drop(patient_index, axis=0)
            neg_color = 'rgb(0, 0, 255)'
            pos_color = 'rgb(255, 0, 0)'

        show_legend = False

        model = LogisticRegression()

        # predictions
        pred = decision_bounds(models[model_index]['scale_x'],
                               models[model_index]['scale_y'],
                               models[model_index]['features'],
                               'glaucoma',
                               model,
                               df0)
        models[model_index]['pred'] = pred
        xx, yy = np.meshgrid(models[model_index]['scale_x'], models[model_index]['scale_y'])

        trace_1 = go.Scattergl(
                    x=xx.ravel()[models[model_index]['pred'] == 0],
                    y=yy.ravel()[models[model_index]['pred'] == 0],
                    mode='markers',
                    marker={
                        'color': models[model_index]['pred'][models[model_index]['pred'] == 0],
                        'colorscale': [[0, neg_color], [1, pos_color]],
                        'cmin': 0,
                        'cmax': 1
                    },
                    opacity=1,
                    name='negative prediction',
                    showlegend=show_legend
                )

        trace_2 = go.Scattergl(
                    x=xx.ravel()[models[model_index]['pred'] == 1],
                    y=yy.ravel()[models[model_index]['pred'] == 1],
                    mode='markers',
                    marker={
                        'color': models[model_index]['pred'][models[model_index]['pred'] == 1],
                        'colorscale': [[0, neg_color], [1, pos_color]],
                        'cmin': 0,
                        'cmax': 1
                    },
                    opacity=1,
                    name='positive prediction',
                    showlegend=show_legend
                )

        if patient_index != None:
            trace_3 = go.Scattergl(
                        x=[df.loc[patient_index, models[model_index]['features'][0]]],
                        y=[df.loc[patient_index, models[model_index]['features'][1]]],
                        mode='markers',
                        marker={'symbol': 'cross', 'color': 'rgb(255, 255, 255)', 'size': 12},
                        name='patient data',
                        showlegend=show_legend,
                    )
            data = [trace_1, trace_2, trace_3]
        else:
            data = [trace_1, trace_2]

        layout = go.Layout(
                    autosize=False,
                    margin={'r': 0, 'l': 60, 't': 0, 'b': 70},
                    width=250,
                    height=250,
                    xaxis={'title': models[model_index]['labels'][0],
                           'tickvals': models[model_index]['xtickvals'],
                           'titlefont': {'size': 'auto'},
                           'showgrid': False,
                           'zeroline': False,
                           'showline': True},
                    yaxis={'title': models[model_index]['labels'][1],
                           'tickvals': models[model_index]['ytickvals'],
                           'titlefont': {'size': 'auto'},
                           'showgrid': False,
                           'zeroline': False,
                           'showline': True}
                )

        return go.Figure(data=data, layout=layout)
    return callback

# -------------------
# Initialize the app
# -------------------
app = dash.Dash()

# -----------------
# create the layout
# -----------------
app.layout = html.Div(children=[
    # Glaucoma Net logo
    html.Div([html.H1('GlaucomaNet')],
        className='banner',
        style={'background': 'rgb(230,230,230)',
               'marginLeft': '20px',
               'marginTop': '10px',
               'marginBottom': '-10px',
               'font-family': 'Russo One',
               'color': '#4D637F',
               'borderRadius': '25px'}),
    # container for Patient Data
    html.Div([
        # Select patient, show age, sex, race
        html.H2(dcc.Markdown(''' **Patient Data** '''),
                style={'fontFamily': 'Avenir Next',
                       'color': '#869bb7',
                       'marginLeft': left_margin,
                       'marginTop': '15px',
                       'marginBottom': '0px',
                       'display': 'inline-block'}),
        html.Div([
            # dropdown for selecting patient
            html.Div([
                html.Label('Select Patient', style = {'marginLeft': left_margin}),
                html.Div([
                        dcc.Dropdown(
                            id='patient_dropdown',
                            options=dropdown_opts,
                            value=None
                        ),
                    ],
                    style={'width': '80%', 'marginLeft': left_margin}
                ),
            ], className='three columns'),
            # show age, sex, race
            html.Div([
                html.Div([
                        html.Div([
                            html.H4('''Age:''')
                        ]),
                        html.Div([
                            html.H4('''Sex:''')
                        ]),
                        html.Div([
                            html.H4('''Race:''')
                        ]),
                    ],
                    style={'width': '3%'},
                    className='two columns'
                ),
                html.Div([
                        html.Div([
                            html.H4(id='Age')
                        ]),
                        html.Div([
                            html.H4(id='Sex')
                        ]),
                        html.Div([
                            html.H4(id='Race')
                        ]),
                    ],
                    style={'width': '25%'},
                    className='three columns'
                ),
            ],className='row'),
        ],className='row'),

        # table with patient data
        html.Div(id='patient_data', style={'marginLeft': left_margin}),
    ],
    style={'marginBottom': 20,
           'marginTop': 20,
           'marginLeft':20,
           'marginRight': 20,
           'height': '340px',
           'background': 'white',
           'borderRadius': '25px',
          }
    ),

    # container for the predicted diagnosis and probability
    html.Div([
        # show the predicted diagnosis
        html.Div([
            html.Div([
                html.H2(dcc.Markdown(''' **Predicted Diagnosis:** '''), style=header_style),
            ], className='six columns'),
            html.Div(id='prediction', className='two columns'),
        ], className='row'),

        # show posterior probability
        html.Div([
            html.Div([
                html.H2(dcc.Markdown(''' **Probability of a Positive Diagnosis:** '''), style=header_style),
            ], className='six columns'),
            html.Div(id='probability', className='two columns'),
        ], className='row'),
    ],
    style = {'marginBottom': 20,
             'marginTop': 20,
             'marginLeft': 20,
             'marginRight': 20,
             'height': '170px',
             'background': 'white',
             'borderRadius': '25px',
             }
    ),
    # container for the model plots
    html.Div([
        # show model plots
        html.H2(dcc.Markdown(''' **Model Predictions** '''),
                style={'fontFamily': 'Avenir Next',
                       'color': '#869bb7',
                       'marginLeft': left_margin,
                       'marginTop': '15px',
                       'marginBottom': '0px',
                       'display': 'inline-block'}),
        # model plots
        html.Div([
            html.Div([
                dcc.Graph(id='m1')],
                className='three columns',
                style={'width': 200, 'marginRight': 40}),

            html.Div([
                dcc.Graph(id='m2')],
                className='three columns',
                style={'width': 200, 'marginRight': 40}),

            html.Div([
                dcc.Graph(id='m3')],
                className='three columns',
                style={'width': 200, 'marginRight': 40}),

            html.Div([
                dcc.Graph(id='m4')],
                className='three columns',
                style={'width': 200}),
        ], className='row', style={'marginLeft': 20}),
        # container for the legend
        html.Div([
            # NEGATIVE
            html.Div(style={'background': 'rgb(0,0,255)',
                            'width': '40px',
                            'height': '20px',
                            'marginLeft': '20px'},
                     className='one columns'),
            html.Div(['Negative Prediction'],
                    className='four columns',
                     style={'marginLeft': '5px',
                            'width': '150px'}),
            # POSITIVE
            html.Div(style={'background': 'rgb(255,0,0)',
                           'width': '40px',
                            'height': '20px',
                            'marginLeft': '20px'},
                     className='one columns'),
            html.Div(['Positive Prediction'],
                     className='four columns',
                     style={'marginLeft': '5px',
                            'width': '150px'}),
            html.Div([dcc.Markdown([''' ### + ### '''])],
                     style={'width': '40px',
                            'height': '20px',
                            'marginLeft': '10px',
                            'marginTop': '-27px'},
                     className='one columns'),
            html.Div(['Patient Data'],
                     className='four columns',
                     style={'marginLeft': '35px',
                            'width': '150px'}),
        ], className='row')
    ],
    style = {'marginBottom': 20,
             'marginTop': 20,
             'marginLeft': 20,
             'marginRight': 20,
             'height': '365px',
             'background': 'white',
             'borderRadius': '25px',
             }
    ),

], style={'marginBottom': 0,
          'marginTop': 0,
          'marginLeft': 0,
          'marginRight': 0,
          'width':'1300px',
          'position': 'absolute',
          'background': 'rgb(230,230,230)',
          'borderRadius': '25px'}
)

# style docs
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.css.append_css({
    "external_url": "//fonts.googleapis.com/css?family=Bungee"})
app.css.append_css({
    "external_url": "//fonts.googleapis.com/css?family=Russo One"})
app.css.append_css({
    "external_url": "//fonts.googleapis.com/css?family=Dosis:Medium"})

# ----------------------
# display age, sex, race
# ----------------------
@app.callback(
    dash.dependencies.Output('Age', 'children'),
    [dash.dependencies.Input('patient_dropdown', 'value')])
def get_age(row):
    if row == None:
        return
    row = int(row)
    return str(int(np.floor(df.loc[row, 'age'])))

@app.callback(
    dash.dependencies.Output('Sex', 'children'),
    [dash.dependencies.Input('patient_dropdown', 'value')])
def get_age(row):
    if row == None:
        return
    row = int(row)
    if df.loc[row, 'male']:
        return 'Male'
    else:
        return 'Female'

@app.callback(
    dash.dependencies.Output('Race', 'children'),
    [dash.dependencies.Input('patient_dropdown', 'value')])
def get_age(row):
    if row == None:
        return
    row = int(row)
    return {
        '1': 'European descent',
        '2': 'Asian descent',
        '3': 'African descent',
        '4': 'Hispanic',
        'NA': 'NA'
    }[str(int(df.loc[row, 'race']))]


# ----------------------------------------------
# create a table with the current patient's data
# ----------------------------------------------
@app.callback(
    dash.dependencies.Output('patient_data', 'children'),
    [dash.dependencies.Input('patient_dropdown', 'value')])
def generate_table(row):
    columns = df.columns[~df.columns.isin(['glaucoma', 'age', 'race', 'male'])]
    labels = ['CDR', 'IOP', 'AVGT', 'GHT', 'VFI', 'MD', 'PSD', 'CDR_AVGT', 'GHT_PSD']
    if row == None:
        return html.Table([html.Tr([html.Th(label) for label in labels])])
    else:
        row = int(row)
        # get the current patient's data and format for table
        df_this_patient = df.iloc[row]
        df_this_patient['cdr_avgt'] = np.round(df_this_patient['cdr_avgt'], 3)
        df_this_patient['ght_psd'] = np.round(df_this_patient['ght_psd'], 3)
        return html.Table(
            # Header
            [html.Tr([html.Th(label) for label in labels])] +
            # Body
            [html.Tr([
                html.Td(df_this_patient[col]) for col in columns])]
    )

# -------------------------------
# display the predicted diagnosis
# -------------------------------
@app.callback(
    dash.dependencies.Output('prediction', 'children'),
    [dash.dependencies.Input('patient_dropdown', 'value')])
def get_prediction(row):
    if row == None:
        return
    row = int(row)
    X_train = df.drop(row, axis=0).drop('glaucoma', axis=1).values
    Y_train = df['glaucoma'].drop(row, axis=0).values
    X_test = df.loc[row, df.columns[~df.columns.isin(['glaucoma'])]]
    # psd and cdr_avgt
    pipe1 = make_pipeline(ColumnSelector(cols=(9, 10)), LogisticRegression())
    # ght and cdr_avgt
    pipe2 = make_pipeline(ColumnSelector(cols=(6, 10)), LogisticRegression())
    # ght_psd and cdr_avgt
    pipe3 = make_pipeline(ColumnSelector(cols=(11, 10)), LogisticRegression())
    # cdr psd
    pipe4 = make_pipeline(ColumnSelector(cols=(3, 9)), LogisticRegression())
    est = [('lr1', pipe1), ('lr2', pipe2), ('lr3', pipe3), ('lr4', pipe4)]
    model = VotingClassifier(estimators=est, voting='hard')
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test.values.reshape(1, -1))
    if int(y_pred) == 0:
        return html.H2(dcc.Markdown(''' **Negative** '''), style={'color': 'rgb(0,0,255)'})
    else:
        return html.H2(dcc.Markdown(''' **Positive** '''), style={'color': 'rgb(255,0,0)'})



# -----------------------------------------
# display probability of positive diagnosis
# -----------------------------------------
@app.callback(
    dash.dependencies.Output('probability', 'children'),
    [dash.dependencies.Input('patient_dropdown', 'value')])
def get_probability(patient_index):
    if patient_index == None:
       return
    patient_index = int(patient_index)
    feature_list = [['cdr_avgt', 'psdonl'],
                    ['cdr_avgt', 'ght'],
                    ['cdr_avgt', 'ght_psd'],
                    ['cdratio', 'psdonl']]
    X_train = df.drop(patient_index, axis=0)
    Y_train = df['glaucoma'].drop(patient_index, axis=0)
    X_test = df.loc[patient_index]
    # classifier performance
    hit_rate = .73
    tneg_rate = .72
    # get probabilities from the models
    all_probs = []
    for features in feature_list:
        model = LogisticRegression()
        model.fit(X_train[features].values, Y_train.values)
        probs = model.predict_proba(X_test[features].values.reshape(1, -1))
        all_probs.append(probs[0][1])
    all_probs = np.array(all_probs)
    # get the majority probabilities
    if sum(all_probs > .5) > sum(all_probs < .5):
        maj_probs = all_probs[all_probs > .5]
        # probabilities given positive prediction
        p1 = hit_rate
        p2 = 1 - tneg_rate
    else:
        maj_probs = all_probs[all_probs < .5]
        p1 = 1 - hit_rate
        p2 = tneg_rate
    # get posterior probability of a positive diagnosis given the data and
    # classifier performance
    p_pos = get_prob(maj_probs, p1, p2)
    # output the probability as a string
    return html.H2(str(np.round(p_pos, 3)))

# -------------------
# display model plots
# -------------------
app.callback(
    dash.dependencies.Output('m1', 'figure'),
    [dash.dependencies.Input('patient_dropdown', 'value')]
)(do_model_plots('cdr_avgt | psd'))

app.callback(
    dash.dependencies.Output('m2', 'figure'),
    [dash.dependencies.Input('patient_dropdown', 'value')]
)(do_model_plots('cdr_avgt | ght'))

app.callback(
    dash.dependencies.Output('m3', 'figure'),
    [dash.dependencies.Input('patient_dropdown', 'value')]
)(do_model_plots('cdr_avgt | ght_psd'))

app.callback(
    dash.dependencies.Output('m4', 'figure'),
    [dash.dependencies.Input('patient_dropdown', 'value')]
)(do_model_plots('cdr | psd'))


if __name__ == '__main__':
    app.run_server(debug=True)


