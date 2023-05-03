import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, dcc, dash_table
import io
import datetime
import base64
from dash.dependencies import Input, Output, State
from DataTypeInference import obtain_feature_type_table, createDatasetObject
import arff
import settings
import dash_mantine_components as dmc
import duplicates_and_missing
import type_integrity
import outliers_and_correlations
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import label_purity
import helper_and_transform_functions
import testingFile
#df = pd.read_csv('datasets\Iris.csv')
sortingHatInf_datatypes = ['not-generalizable', 'floating', 'integer', 'categorical', 'boolean', 'datetime', 'sentence', 'url',
                           'embedded-number', 'list', 'context-specific', 'numeric']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

button_style = {'background-color': 'blue',
                    'color': 'white',
                    'height': '50px',
                    'margin-top': '50px',
                    'margin-left': '50px'}
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Data quality toolkit"

app.layout = html.Div([
    html.Div("Data quality toolkit", style={'fontSize':50, 'textAlign':'center'}),
    html.Hr(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        max_size=-1,
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-div'),
    # html.Div(id='dd-output-container'),
    html.Div(id='output-data-upload'),





])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if '.csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif '.xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif '.arff' in filename:
            data = arff.load(io.StringIO(decoded.decode('utf-8')))
            columns = [attr[0] for attr in data['attributes']]
            df = pd.DataFrame(data['data'], columns=columns)
        featureTypeTable = obtain_feature_type_table(df)
        #settings.first_time = True
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.P('Uploaded file: {}'.format(filename)),
        html.P('Choose your target column'),
        dcc.Dropdown(id="targetColumn", options=[{'label':'None', 'value':'None'}] + [{'label':x, 'value':x} for x in df.columns], value = 'None'),
        #html.Button(id='submit-button', children='Run checks'),
        html.Div(id='dd-output-container'),

        #Feature type table
        dash_table.DataTable(
            id='table-dropdown',
            data = featureTypeTable.to_dict('records'),
            columns = [{"name": i, "id": i, 'presentation': 'dropdown'} for i in featureTypeTable.columns],
            editable=True,
            dropdown={
                i: {'options': [{'label': j, 'value': j} for j in
                                       sortingHatInf_datatypes]} for i in
                featureTypeTable.columns},
            # style_table={
            #     'overflowX': 'scroll'}
        ),
        html.Button('Run data quality checks', id='run-checks-button', n_clicks=0, style=button_style),


        html.Div(id='container-checks-button-pressed'),

        #Dataset overview section
        html.Hr(),  # horizontal line
        html.H6('Dataset overview', style={'textAlign': 'center'}),
        html.P('Click the dataset to edit a cell, press the EXPORT button to download the edited dataset',
               style={'textAlign': 'center'}),
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            page_size = 10,
            editable=True,
            export_format='csv',
            style_table={
                'overflowX': 'scroll'
            }
        ),
        dcc.Store(id='stored-data', data = df.to_dict('records'), storage_type='memory'),
        html.Hr(),  # horizontal line
    ])
@app.callback(
    [Output('dd-output-container', 'children'), Output('output-div', 'children')],
    [Input('targetColumn', 'value')]#, Input('submit-button','n_clicks'), State('targetColumn','targetvalue')]
)
def update_output(value):#, n, target):
    return f'You have selected {value}, as your target column', dash.no_update


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(
    Output('container-checks-button-pressed', 'children'),
    [Input('run-checks-button', 'n_clicks'),
     Input('stored-data', 'data'),
     Input('table-dropdown', 'data'),
     Input('targetColumn', 'value')]
)
def run_checks(n_clicks, df_json, dtypes, target_column):#, n, target):
    if n_clicks >= 1: #run checks button clicked
        dtypes_dict = dtypes[0]
        df = pd.DataFrame(df_json) #deserialize JSON string stored in web browser
        ds = createDatasetObject(df, dtypes_dict, target_column)
        #TODO: convert to Deepchecks datasets, keep in mind length error

        #Running of the checks
        #duplicates & missing
        df_missing_values = duplicates_and_missing.missing_values(df)
        df_duplicates = duplicates_and_missing.duplicates(df)     #TODO: duplicates check testen

        #type integrity checks
        df_amount_of_diff_values = type_integrity.amount_of_diff_values(df)
        df_mixed_data_types = type_integrity.mixed_data_types(df)
        df_special_characters = type_integrity.special_characters(df)
        df_string_mismatch = type_integrity.string_mismatch(df)

        #outliers & correlations
        df_feature_feature_correlation, correlationFig = outliers_and_correlations.feature_feature_correlation(ds)
        df_outliers, amount_of_outliers, threshold = outliers_and_correlations.outlier_detection(ds)
        pandas_dq_report = helper_and_transform_functions.pandas_dq_report(df, target_column)

        if target_column != 'None': #target column supplied
            df_feature_label_correlation = outliers_and_correlations.feature_label_correlation(ds)

            #label purity checks
            df_class_imbalance = label_purity.class_imbalance(ds)
            df_conflicting_labels, percent_conflicting = label_purity.conflicting_labels(ds)
        else: #no target column selected
            df_feature_label_correlation = pd.DataFrame({
                "Message": ["This check is not applicable as there is no target column selected"]})
            df_conflicting_labels = pd.DataFrame(
                {"Message": ["This check is not applicable as there is no target column selected"]})
            percent_conflicting = 0
            df_class_imbalance = pd.DataFrame(
                {"Message": ["This check is not applicable as there is no target column selected"]})

        return html.Div([#Data issue / check results section
                html.Hr(),  # horizontal line
                html.Hr(),  # horizontal line
                html.H6('Profiling report and issue overview', style={'textAlign':'center'}),
                html.P('This section contains a profling report showing important information'
                       ' regarding ML issues found in the dataset', style={'textAlign':'center'}),
                #TODO: TQDM loader
                #TODO: warning reports + general profiling section (o.b.v. data readiness report?)
                #dash_table.DataTable(id='table', columns=[{"name": i, "id": i} for i in pandas_dq_report.columns], data=df.to_dict('records')),
                dash_table.DataTable(pandas_dq_report.to_dict('records')),

                dmc.Accordion(
                children=[
                    dmc.AccordionItem(
                        [
                            dmc.AccordionControl("Duplicates & missing values ({})".format(36)),
                            dmc.AccordionPanel([dash_table.DataTable(df_missing_values.to_dict('records')),
                                                dash_table.DataTable(df_duplicates.to_dict('records'))
                                               ]
                            ),

                        ],
                        value="duplicatesandmissing",
                    ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Type integrity ({})".format(36)),
                        dmc.AccordionPanel([html.H6('Amount of distinct values per column', style={'textAlign':'center'}),
                                            html.P('Checks the amount of different values for each column',
                                                    style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_amount_of_diff_values.to_dict('records')),
                                            #dash_table.DataTable(df_mixed_data_types.to_dict('records')), #TODO: fix error
                                            html.H6("Special characters check", style={'textAlign': 'center'}),
                                            html.P("checks for characters like '?!$^&#' ", style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_special_characters.to_dict('records')),
                                            html.H6("String mismatch / cell entity check",
                                                    style={'textAlign': 'center'}),
                                            html.P("Checks for strings that have the same base form, like 'red', 'Red', 'RED!' (base form 'red' )",
                                                   style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_string_mismatch.to_dict('records'))]
                        ),
                    ],
                    value="typeintegrity",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Outliers & correlations ({})".format(36)),
                        dmc.AccordionPanel([html.H6("Outlier samples check",
                                                    style={'textAlign': 'center'}),
                                            html.P(["Function that checks for outliers samples (jointly across all features) using "
                                                   "the LoOP algorithm: ", html.A('LoOp paper', href='https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf')],
                                                   style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_outliers.to_dict('records'), style_table={'overflowX': 'scroll'}),
                                            html.P("{} outliers have been found above the set probability threshold of {}".format(amount_of_outliers, threshold),
                                                   style={'textAlign': 'center'}),
                                            html.H6("Feature-feature correlation check",
                                                    style={'textAlign': 'center'}),
                                            html.P("computes the correlation between each feature pair;"
                                                   " Methods to calculate for each feature label pair:"
                                                   " numerical-numerical: Pearson’s correlation coefficient"
                                                   " numerical-categorical: Correlation ratio"
                                                   " categorical-categorical: Symmetric Theil’s U",
                                                   style={'textAlign': 'center'}),
                                            #dash_table.DataTable(df_feature_feature_correlation.to_dict('records')),
                                            dcc.Graph(figure=correlationFig),
                                            html.H6("Feature-label correlation check", style={'textAlign': 'center'}),
                                            html.P("Computes the correlation between each feature and the label, "
                                                   "in a similar fashion as the feature-feature correlation",
                                                   style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_feature_label_correlation.to_dict('records'))]
                        ),
                    ],
                    value="outliersandcorrelations",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Label purity ({})".format(36)),
                        dmc.AccordionPanel([html.H6('Class imbalance check', style={'textAlign':'center'}),
                                            html.P('Checks to what extent the amount of instances per label value are equal',
                                                    style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_class_imbalance.to_dict('records')),
                                            html.H6('Conflicting labels check',
                                                    style={'textAlign': 'center'}),
                                            html.P('Checks for instances with exactly the same feature values, but different labels (which confuse the model)',
                                                   style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_conflicting_labels.to_dict('records')),
                                            html.P("There are {}% conflicting labels".format(percent_conflicting), style={'textAlign': 'center'})

                        ]
                        ),
                    ],
                    value="labelpurity",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Bias & fairness ({})".format(36)),
                        dmc.AccordionPanel(
                            "Configure temp appearance and behavior with vast amount of settings or overwrite any part of "
                            "component styles "
                        ),
                    ],
                    value="biasandfairness",
                ),
            ],
        )
        ]
        )
    else:
        html.Hr()


if __name__ == '__main__':
    app.run_server(debug=True)
