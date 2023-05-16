import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, dcc, dash_table
import io
import datetime
import base64
from dash.dependencies import Input, Output, State



import arff
from sklearn.preprocessing import LabelEncoder
import dash_mantine_components as dmc

from DataTypeInference import obtain_feature_type_table, createDatasetObject
import fairness_checks
import duplicates_and_missing
import type_integrity
import outliers_and_correlations
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import label_purity
import plot_and_transform_functions
import testingFile
#df = pd.read_csv('datasets\Iris.csv')
sortingHatInf_datatypes = ['not-generalizable', 'floating', 'integer', 'categorical', 'boolean', 'datetime', 'sentence', 'url',
                           'embedded-number', 'list', 'context-specific', 'numeric']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']#[dbc.themes.SUPERHERO]#

button_style = {'background-color': 'blue',
                    'color': 'white',
                    'height': '50px',
                    'margin-top': '50px',
                    'margin-left': '50px'}
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
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
    html.Div(id='output-data-upload'),
    #dcc.Graph(figure=fig)





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
            id='dtypes_dropdown',
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
        html.Hr(),  # horizontal line


        dcc.Loading(children=html.Div(id='container-checks-button-pressed'), type =  'cube'),
        dcc.Loading(children=html.Div(id='long_running_time_accordion'), type='circle'),

        #Dataset overview section
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
        dcc.Store(id='stored-data', data = df.to_dict('records'), storage_type='memory'), #TODO: aanpassen pickle oid?
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
     State('stored-data', 'data'),
     State('dtypes_dropdown', 'data'),
     State('targetColumn', 'value')]
)
def run_checks(n_clicks, df_json, dtypes, target):#, n, target):
    if n_clicks >= 1: #run checks button clicked
        dtypes_dict = dtypes[0]
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@dtypes_dict', dtypes_dict)
        df = pd.DataFrame(df_json) #deserialize JSON string stored in web browser
        ds = createDatasetObject(df, dtypes_dict, target)
        #TODO: convert to Deepchecks datasets, keep in mind length error

        #Running of the checks
        #duplicates & missing
        df_missing_values = duplicates_and_missing.missing_values(df)
        df_duplicates = duplicates_and_missing.duplicates(df)     #TODO: duplicates check testen
        df_duplicate_columns = duplicates_and_missing.duplicate_column(df)

        #type integrity checks
        df_amount_of_diff_values = type_integrity.amount_of_diff_values(df)
        df_mixed_data_types = type_integrity.mixed_data_types(df)
        df_special_characters = type_integrity.special_characters(df)
        #print(df_special_characters)
        df_string_mismatch = type_integrity.string_mismatch(df)

        #outliers & correlations
        df_feature_feature_correlation, correlationFig = outliers_and_correlations.feature_feature_correlation(ds)
        df_outliers, amount_of_outliers, threshold = outliers_and_correlations.outlier_detection(ds)
        #df_outlier_per_column = pd.DataFrame({"Message": ["No outliers per column detected"]})
        pandas_dq_report = plot_and_transform_functions.pandas_dq_report(df, target)

        #the encoded dataframe
        df = plot_and_transform_functions.clean_dataset(df)
        encoded_dataframe, mapping_encoding = plot_and_transform_functions.encode_categorical_columns(df, target, dtypes_dict)

        #plots
        distribution_figures = plot_and_transform_functions.plot_dataset_distributions(df, dtypes_dict) #list of all column data distribution figures
        data_distribution_figures_div = html.Div([dcc.Graph(id='multi_' + str(i), figure=distribution_figures[i], style={'display': 'inline-block', 'width': '30vh', 'height': '30vh'}) for i in range(len(distribution_figures))])
        #missingno_plot = plot_and_transform_functions.missingno_plot(df)
        pcp_plot = plot_and_transform_functions.pcp_plot(encoded_dataframe, target) #TODO: probleem; grote datasets encoding tript, target encoden?


        if target != 'None': #target column supplied
            df_feature_label_correlation = outliers_and_correlations.feature_label_correlation(ds)

            #label purity checks
            df_class_imbalance, _ = label_purity.class_imbalance(ds)
            df_conflicting_labels, percent_conflicting = label_purity.conflicting_labels(ds)
        else: #no target column selected
            df_feature_label_correlation = pd.DataFrame({
                "Message": ["This check is not applicable as there is no target column selected"]})
            df_conflicting_labels = pd.DataFrame(
                {"Message": ["This check is not applicable as there is no target column selected"]})
            percent_conflicting = 0
            df_class_imbalance = pd.DataFrame(
                {"Message": ["This check is not applicable as there is no target column selected"]})
            fig_class_imbalance = html.Div()

        return html.Div([#Data issue / check results section
                html.H6('Profiling report and issue overview', style={'textAlign':'center'}),
                html.P('This section contains a profling report showing important information'
                       ' regarding ML issues found in the dataset', style={'textAlign':'center'}),
                #TODO: TQDM loader
                #TODO: warning reports + general profiling section (o.b.v. data readiness report?)

                dash_table.DataTable(pandas_dq_report.to_dict('records'), style_table={
                                                                            'overflowX': 'scroll'
                                                                        }),

                dmc.Accordion(
                children=[
                    dmc.AccordionItem(
                        [
                            dmc.AccordionControl("Duplicates & missing values ({})".format(36)),
                            dmc.AccordionPanel([html.H6('Missing values check', style={'textAlign':'center'}),
                                                html.P('Checks the type and amount of missing values. The potential total missingness is the'
                                                       ' percent missing plus the percentage of zeros.',
                                                    style={'textAlign': 'center'}),
                                                dash_table.DataTable(df_missing_values.to_dict('records')),
                                                html.Hr(),
                                                html.H6('Duplicates check', style={'textAlign': 'center'}),
                                                html.P('Checks whether there are any duplicates and displays the row numbers of the  duplicate instances.',
                                                    style={'textAlign': 'center'}),
                                                dash_table.DataTable(df_duplicates.to_dict('records')),
                                                html.H6('Duplicate columns check', style={'textAlign': 'center'}),
                                                html.P(
                                                    'Checks whether there are any exact duplicate columns (which slows down the training time of your ML model).',
                                                    style={'textAlign': 'center'}),
                                                dash_table.DataTable(df_duplicate_columns.to_dict('records')),
                                                #dcc.Graph(figure=mpl_to_plotly(missingno_plot))
                                               ]
                            ),

                        ],
                        value="duplicatesandmissing",
                    ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Type integrity ({})".format(36)),
                        dmc.AccordionPanel([html.H6('Amount of distinct values per column', style={'textAlign':'center'}),
                                            html.P('Checks the amount of different values for each column.',
                                                    style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_amount_of_diff_values.to_dict('records')),
                                            html.Hr(),

                                            html.H6('Mixed data types check',
                                                    style={'textAlign': 'center'}),
                                            html.P('Checks for different data types in your dataset, and displays some random samples.',
                                                   style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_mixed_data_types.to_dict('records')), #TODO: fix error
                                            html.Hr(),

                                            html.H6("Special characters check", style={'textAlign': 'center'}),
                                            html.P("checks for characters like '?!$^&#'.", style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_special_characters.to_dict('records')),
                                            html.Hr(),
                                            html.H6("String mismatch / cell entity check",
                                                    style={'textAlign': 'center'}),
                                            html.P("Checks for strings that have the same base form, like 'red', 'Red', 'RED!' (base form 'red' ).",
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
                                                   "the LoOP algorithm: ", html.A('LoOp paper.', href='https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf')],
                                                   style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_outliers.to_dict('records'), style_table={'overflowX': 'scroll'}),
                                            html.P("{} outliers have been found above the set probability threshold of {}.".format(amount_of_outliers, threshold),
                                                   style={'textAlign': 'center'}),
                                            dcc.Graph(figure=pcp_plot),
                                            html.Hr(),
                                            html.H6("Feature-feature correlation check",
                                                    style={'textAlign': 'center'}),
                                            html.P("computes the correlation between each feature pair;"
                                                   " Methods to calculate for each feature label pair:"
                                                   " numerical-numerical: Pearson’s correlation coefficient"
                                                   " numerical-categorical: Correlation ratio"
                                                   " categorical-categorical: Symmetric Theil’s U.",
                                                   style={'textAlign': 'center'}),
                                            #dash_table.DataTable(df_feature_feature_correlation.to_dict('records')),
                                            dcc.Graph(figure=correlationFig),
                                            html.Hr(),
                                            html.H6("Feature-label correlation check", style={'textAlign': 'center'}),
                                            html.P("Computes the correlation between each feature and the label, "
                                                   "in a similar fashion as the feature-feature correlation.",
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
                                            html.P('Checks the distribution of instances per label.',
                                                    style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_class_imbalance.to_dict('records')),#TODO: checken hoe dit zit bij regression
                                            html.Hr(),
                                            html.H6('Conflicting labels check',
                                                    style={'textAlign': 'center'}),
                                            html.P('Checks for instances with exactly the same feature values, but different labels (which can confuse your ML model).',
                                                   style={'textAlign': 'center'}),
                                            dash_table.DataTable(df_conflicting_labels.to_dict('records')),
                                            html.P("There are {}% conflicting labels.".format(percent_conflicting), style={'textAlign': 'center'}),



                        ]
                        ),
                    ],
                    value="labelpurity",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Bias & feature information ({})".format(36)),
                        dmc.AccordionPanel([
                            #Data distribution plots
                            html.H6('Data distribution', style={'textAlign': 'center'}),
                            # data distribution plots
                            html.P(
                                'The following plots give insights into the datasets central tendency and spread. Each plot represents a variables distribution,'
                                ' with the x-axis showing its value and the y-axis indicating the frequency/proportion of data points with that value.',
                                style={'textAlign': 'center'}),

                            data_distribution_figures_div,
                            html.Hr(),

                            #Feature importance plot
                            html.H6('Feature importance analysis', style={'textAlign': 'center'}),
                            html.P('Displays the feature importance based on target encoded values.',
                                style={'textAlign': 'center'}),
                            dcc.Loading(
                            id="loading-2",
                            children=html.Div(id="feature_importance_plot_div")),
                            html.Hr(),

                            #Subgroup bias analysis
                            html.H6('Bias analysis', style={'textAlign': 'center'}),
                            html.P(
                                'In the dropdown menu below, select the sensitive features (if any) that exist in your dataset. The bar chart indicates the amount of times'
                                'a specific sensitive subgroup appears in your dataset, and the distribution of the class label for that subgroup'
                                'NOTE: it can take a while in order for the chart to be displayed.',
                                style={'textAlign': 'center'}),
                            dcc.Dropdown(id="biasDropdown",
                                         options=[{'label': x, 'value': x} for x in df.columns], multi=True,
                                         placeholder="Select sensitive feature(s)"),
                            dcc.Loading(
                                id="loading-3",
                                children=html.Div(id="biasGraph")),
                        ]),
                    ],
                    value="biasandfairness",
                ),
            ],
        )
        ]
        )
    else:
        html.Hr()


# @app.callback(
#     [Output('cleanlab_table', 'children'),Output('amount_of_label_errors', 'children')],
#     [Input('run-checks-button', 'n_clicks'),
#      State('stored-data', 'data'),
#      State('dtypes_dropdown', 'data'),
#      State('targetColumn', 'value')]
# )
# def cleanlab_label_error_check(n_clicks, df_json, dtypes, target):#, n, target):
#     if n_clicks >= 1: #run checks button clicked
#         if target != 'None':
#             df = pd.DataFrame(df_json)
#             dtypes_dict = dtypes[0]
#             encoded_dataframe, mapping_encoding = plot_and_transform_functions.encode_categorical_columns(df, target,
#                                                                                                           dtypes_dict)
#             _, issues_dataframe_only_errors, wrong_label_count = label_purity.cleanlab_label_error(encoded_dataframe, target)
#             return dash_table.DataTable(issues_dataframe_only_errors.to_dict('records')), f'Cleanlab detected {wrong_label_count} potential label errors'

@app.callback(
    Output('feature_importance_plot_div', 'children'),
    [Input('run-checks-button', 'n_clicks'),
     State('stored-data', 'data'),
     State('dtypes_dropdown', 'data'),
     State('targetColumn', 'value')]
)
def feature_importance_plot(n_clicks, df_json, dtypes, target):#, n, target):
    if n_clicks >= 1: #run checks button clicked

        if target != 'None':
            df = pd.DataFrame(df_json)
            df = plot_and_transform_functions.clean_dataset(df)
            dtypes_dict = dtypes[0]
            feature_importance_plot = plot_and_transform_functions.plot_feature_importance(df, target, dtypes_dict)
            return dcc.Graph(figure = feature_importance_plot)


@app.callback(
    Output(component_id='biasGraph', component_property='children'),
    [Input(component_id='biasDropdown', component_property='value'),
     State('stored-data', 'data'),
     State('targetColumn', 'value')])

def bias_graph(bias_dropdown, df_json, target):
    if bias_dropdown:
        if target in bias_dropdown:
            bias_dropdown = bias_dropdown.remove(target) # as this just gives the class distribution, which is already presented before, and hinders logical results when
        # combined with other features
        if bias_dropdown:
            # then atleast one sensitive feature that is not the target is selected, plot the analysis
            df = pd.DataFrame(df_json)
            df = plot_and_transform_functions.clean_dataset(df)
            sensitive_feature_combinations_table = fairness_checks.sensitive_feature_combinations(df,bias_dropdown, target, bins=5)
            stacked_bar_chart_fig = fairness_checks.plot_stacked_barchart(sensitive_feature_combinations_table)
            return dcc.Graph(figure=stacked_bar_chart_fig)



@app.callback(
    Output('long_running_time_accordion', 'children'),
    [Input('run-checks-button', 'n_clicks'),
     State('stored-data', 'data'),
     State('dtypes_dropdown', 'data'),
     State('targetColumn', 'value')]
)
def cleanlab_label_error_check(n_clicks, df_json, dtypes, target):#, n, target):
    if n_clicks >= 1: #run checks button clicked
        if target != 'None':
            df = pd.DataFrame(df_json)
            dtypes_dict = dtypes[0]
            df = plot_and_transform_functions.clean_dataset(df)
            encoded_dataframe, mapping_encoding = plot_and_transform_functions.encode_categorical_columns(df, target,
                                                                                                          dtypes_dict)
            _, issues_dataframe_only_errors, wrong_label_count = label_purity.cleanlab_label_error(encoded_dataframe, target)
            return dmc.Accordion(
                                children=[
                                    dmc.AccordionItem(
                                        [
                                            dmc.AccordionControl("Extra accordion ({})".format(36)),
                                            dmc.AccordionPanel([#Cleanlab label issues
                                                                html.H6('Cleanlab.ai label issues check', style={'textAlign': 'center'}),
                                                                html.P([
                                                                    "Automatically detects probable label errors using Confident learning, ",
                                                                    html.A('Confident learning paper',
                                                                           href='https://arxiv.org/abs/1911.00068'),
                                                                    ' . The numeric label quality score quantifies clenlabs confidence that the label is correct '
                                                                    '(a low score thus indicates a high probability that the label is wrong).'],
                                                                    style={'textAlign': 'center'}),
                                                                # html.Div(dcc.Loading(id='loading-4',children=html.Div(id="cleanlab_table"))),
                                                                f'Cleanlab detected {wrong_label_count} potential label errors.',
                                                                dash_table.DataTable(issues_dataframe_only_errors.to_dict('records'),
                                                                        page_size = 10,
                                                                        editable=False,
                                                                        style_table={
                                                                            'overflowX': 'scroll'
                                                                        }),
                                                                html.Hr(),
                                                                #Simple model performance
                                                                html.H6('Baseline Performance Assessment', style={'textAlign': 'center'}),
                                                                html.P('Displays the baseline performance of three sklearn models with basic settings.'
                                                                       ' Note that this is in general definitely not the best performance you can achieve,'
                                                                       ' always tune your model parameters (e.g. by performing a gridsearch) to increase model performance.',
                                                                       style={'textAlign': 'center'}),
                                                                dcc.Loading(
                                                                    id="loading-5",
                                                                    children=html.Div(id="baseline_performance_plot")),
                                                                # dcc.Graph(figure=mpl_to_plotly(missingno_plot))
                                                               ]
                                            ),

                                        ],
                                        value="long_running_time_accordion_1",
                                    )])


@app.callback(
    Output('baseline_performance_plot', 'children'),
    [Input('run-checks-button', 'n_clicks'),
     State('stored-data', 'data'),
     State('dtypes_dropdown', 'data'),
     State('targetColumn', 'value')]
)
def baseline_performance_assessment(n_clicks, df_json, dtypes, target):
    if n_clicks >= 1: #run checks button clicked

        if target != 'None':
            df = pd.DataFrame(df_json)
            df = plot_and_transform_functions.clean_dataset(df)
            dtypes_dict = dtypes[0]
            encoded_dataframe, mapping_encoding = plot_and_transform_functions.encode_categorical_columns(df, target,
                                                                                                          dtypes_dict)
            baseline_performance_plot = plot_and_transform_functions.baseline_model_performance(encoded_dataframe, target, dtypes_dict)
            return dcc.Graph(figure = baseline_performance_plot)


if __name__ == '__main__':
    app.run_server(debug=True)
