import pandas as pd
import deepchecks
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, dcc, dash_table
import io
import datetime
import base64
from dash.dependencies import Input, Output, State
# from flask_caching import Cache
import uuid
import time
import scipy.io.arff as arff
from sklearn.preprocessing import LabelEncoder
import dash_mantine_components as dmc
import calculate_dq_label
from DataTypeInference import obtain_feature_type_table, createDatasetObject
import fairness_checks
import Duplicates_and_Missing
import Type_integrity
import outliers_and_correlations
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import label_purity
import plot_and_transform_functions
import testingFile
import os
import logging
deepchecks.set_verbosity(logging.ERROR)
os.environ['DISABLE_LATEST_VERSION_CHECK'] = 'True'

#df = pd.read_csv('datasets\Iris.csv')
#sortingHatInf_datatypes = ['not-generalizable', 'floating', 'integer', 'categorical', 'boolean', 'datetime', 'sentence', 'url', 'embedded-number', 'list', 'context-specific', 'numeric']
sortingHatInf_datatypes = ['not-generalizable', 'categorical', 'boolean', 'datetime', 'sentence', 'url', 'embedded-number', 'list', 'context-specific', 'numeric']

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc_css]#[dbc.themes.SUPERHERO]#['https://codepen.io/chriddyp/pen/bWLwgP.css']#[dbc.themes.SUPERHERO]#

button_style = {'background-color': 'blue',
                    'color': 'white',
                    'height': '50px',
                    'margin-top': '50px',
                    'margin-left': '50px'}
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
current_directory = os.path.dirname(os.path.realpath('dashapp2.py'))
cache_dir = os.path.join(current_directory, 'cached_files')
# Configure the cache
# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': cache_dir  # Directory where cache files will be stored
# })

app.title = "Data quality toolkit"

app.layout = dbc.Container(html.Div([
    html.H1("Data quality toolkit", style={'fontSize':50, 'textAlign':'center'}),
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
    # dcc.Upload(
    #     id='upload-data',
    #     children=html.Div([dbc.Button('Upload a file (.csv, .xls, .xlsx, .arff, .parquet)', color='primary', size='lg')], className="d-grid gap-2 col-6 mx-auto")),
    dcc.Loading(children=html.Div(id='output-data-upload'), type = 'circle', style={'content': "Loadin@@@g..."}),


    #dcc.Graph(figure=fig)





]), fluid=True, className="dbc")

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
    else:
        return dbc.Container(html.Div([html.Br(),html.H6('Welcome to the Data quality toolkit for assessing tabular machine learning datasets,'
                                               ' use the tool by following these four easy steps: ', style={'textAlign': 'left'}), html.Ol(
            children=[
                html.Li('Upload a file (allowed file types: .csv, .xls, .xlsx, .arff, .parquet).'),
                html.Li('Select your target column, and if necessary, correct the automatically inferred data types of your columns.'),
                html.Li("Press 'Run checks', your checks will now be run. (if your dataset is larger than 10,000 rows, 10,000 rows will randomly be sampled)."),
                html.Li("Optional (task dependent): Press 'Run additional checks' to find label errors in your dataset and see the baseline performance on"
                        " three ML models. In addition, you can perform a bias analysis.")

            ], style={'textAlign': 'left'}
        )]), fluid=True, class_name="dbc")

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    filepath = generate_filepath(filename)

    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif '.arff' in filename:
            # Assume that the user uploaded an ARFF file
            data = io.StringIO(decoded.decode('utf-8'))
            arff_data = arff.loadarff(data)
            df = pd.DataFrame(arff_data[0])
            for column in df.columns:
                if df[column].dtype == object:  # Check if the column contains object data type (usually used for strings)
                    df[column] = df[column].str.decode('utf-8', errors='ignore')
        elif '.parquet' in filename:
            # Assume that the user uploaded a Parquet file
            df = pd.read_parquet(io.BytesIO(decoded))

        #Sample 10000 rows if user uploaded a large dataframe
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)

        df.to_pickle(filepath)
        featureTypeTable = obtain_feature_type_table(df)
        #settings.first_time = True
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.',
            html.Br(),
            dbc.Button('DQ label: E', color='danger')
        ], className="d-grid gap-2 col-6 mx-auto")

    return dbc.Container(html.Div([
        html.P('Uploaded file: {}'.format(filename)),
        html.H6('Choose your target column'),
        dcc.Dropdown(id="targetColumn", options=[{'label':'None', 'value':'None'}] + [{'label':x, 'value':x} for x in df.columns], value = 'None'),
        #html.Button(id='submit-button', children='Run checks'),
        html.Div(id='target-selected-container'),
        dcc.Store(id='stored-filepath', data=filepath, storage_type='memory'),

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
            style_table={
                'overflowX': 'scroll', 'height': '250px'}
            # style_table={
            #     'overflowX': 'scroll'}
        ),
        #html.Button('Run checks', id='run-checks-button', n_clicks=0, style=button_style),
        dbc.Button('Run checks', id='run-checks-button', n_clicks=0, color='primary'),
        html.Hr(),  # horizontal line

        html.Div(dcc.Loading(children=html.Div(id='container-checks-button-pressed'), type='cube')),

        #Dataset overview section
        html.H3('Dataset overview', style={'textAlign': 'center'}),
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
        #dcc.Store(id='stored-data', data = df.to_dict('records'), storage_type='memory'), #TODO: aanpassen pickle oid?
        html.Hr(),  # horizontal line
    ]), fluid = True, class_name="dbc")
@app.callback(
    Output('target-selected-container', 'children'),
    Input('targetColumn', 'value')#, Input('submit-button','n_clicks'), State('targetColumn','targetvalue')]
)
def display_selected_target_column(value):#, n, target):
    return f'You have selected {value}, as your target column'




@app.callback(
    Output('container-checks-button-pressed', 'children'),
    [Input('run-checks-button', 'n_clicks'),
     State('stored-filepath', 'data'),
     State('dtypes_dropdown', 'data'),
     State('targetColumn', 'value')]
)
def run_checks(n_clicks, filepath, dtypes, target):#, n, target):
    if n_clicks >= 1: #run checks button clicked
        dtypes_dict = dtypes[0]
        task_type = 'other'
        if target != 'None':
            if dtypes_dict[target] == 'categorical' or dtypes_dict[target] == 'boolean':
                task_type = 'classification'
            else:
                task_type = 'regression'

        df = fetch_data(filepath) #deserialize JSON string stored in web browser
        ds = createDatasetObject(df, dtypes_dict, target)
        #TODO: convert to Deepchecks datasets, keep in mind length error

        #Running of the checks
        check_results = {}
        #duplicates & missing
        df_missing_values = Duplicates_and_Missing.missing_values(df)
        check_results['df_missing_values'] = df_missing_values
        df_missing_values.to_csv('datasets/dataframes_label/df_missing_values.csv')
        #missingno_plot_src = Duplicates_and_Missing.missingno_plot(df)
        df_duplicates = Duplicates_and_Missing.duplicates(df, dtypes_dict)     #TODO: duplicates check testen
        check_results['df_duplicates'] = df_duplicates
        df_duplicates.to_csv('datasets/dataframes_label/df_duplicates.csv')
        df_duplicate_columns = Duplicates_and_Missing.duplicate_column(df)
        check_results['df_duplicate_columns'] = df_duplicate_columns
        df_duplicate_columns.to_csv('datasets/dataframes_label/df_duplicate_columns.csv')

        #type integrity checks
        df_amount_of_diff_values = Type_integrity.amount_of_diff_values(df)
        check_results['df_amount_of_diff_values'] = df_amount_of_diff_values
        df_amount_of_diff_values.to_csv('datasets/dataframes_label/df_amount_of_diff_values.csv')
        df_mixed_data_types = Type_integrity.mixed_data_types(df)
        check_results['df_mixed_data_types'] = df_mixed_data_types
        df_mixed_data_types.to_csv('datasets/dataframes_label/df_mixed_data_types.csv')
        df_special_characters = Type_integrity.special_characters(df)
        check_results['df_special_characters'] = df_special_characters
        df_special_characters.to_csv('datasets/dataframes_label/df_special_characters.csv')
        df_string_mismatch = Type_integrity.string_mismatch(df)
        check_results['df_string_mismatch'] = df_string_mismatch
        df_string_mismatch.to_csv('datasets/dataframes_label/df_string_mismatch.csv')

        #outliers & correlations
        df_feature_feature_correlation, correlationFig = outliers_and_correlations.feature_feature_correlation(ds)
        check_results['df_feature_feature_correlation'] = df_feature_feature_correlation
        df_feature_feature_correlation.to_csv('datasets/dataframes_label/df_feature_feature_correlation.csv')
        df_outliers, amount_of_outliers, threshold = outliers_and_correlations.outlier_detection(ds)
        check_results['df_outliers'] = df_outliers
        df_outliers.to_csv('datasets/dataframes_label/df_outliers.csv')

        #pandas dq report #TODO: vervangen voor eigen completere rapport
        pandas_dq_report = plot_and_transform_functions.pandas_dq_report(df, target)

        #the encoded dataframe
        df = plot_and_transform_functions.clean_dataset(df)
        #encoded_dataframe, mapping_encoding = plot_and_transform_functions.encode_categorical_columns(df, target, dtypes_dict)

        #plots
        # distribution_figures = plot_and_transform_functions.plot_dataset_distributions(df, dtypes_dict) #list of all column data distribution figures
        # data_distribution_figures_div = html.Div([dcc.Graph(id='multi_' + str(i), figure=distribution_figures[i], style={'display': 'inline-block', 'width': '30vh', 'height': '30vh'}) for i in range(len(distribution_figures))])
        #missingno_plot = plot_and_transform_functions.missingno_plot(df)
        label_encoded_df, label_mapping = plot_and_transform_functions.label_encode_dataframe(df, dtypes_dict)
        pcp_plot = plot_and_transform_functions.pcp_plot(label_encoded_df, target)
        box_plot = outliers_and_correlations.box_plot(df, dtypes_dict)

        if task_type == 'classification': #target column supplied
            df_feature_label_correlation = outliers_and_correlations.feature_label_correlation(ds)
            df_feature_label_correlation.to_csv('datasets/dataframes_label/df_feature_label_correlation.csv')
            #label purity checks
            df_class_imbalance, fig_class_imbalance = label_purity.class_imbalance(ds)
            df_class_imbalance.to_csv('datasets/dataframes_label/df_class_imbalance.csv')
            df_conflicting_labels, percent_conflicting = label_purity.conflicting_labels(ds)
            df_conflicting_labels.to_csv('datasets/dataframes_label/df_conflicting_labels.csv')
        else: #no target column selected or not a classification problem, thus not applicable
            df_feature_label_correlation = pd.DataFrame({
                "Check notification": ["This check is not applicable as there is no target column selected or the problem at hand"
                            " is not a classification problem"]})
            df_conflicting_labels = pd.DataFrame(
                {"Check notification": ["This check is not applicable as there is no target column selected or the problem at hand"
                             " is not a classification problem"]})
            percent_conflicting = 0
            df_class_imbalance = pd.DataFrame(
                {"Check notification": ["This check is not applicable as there is no target column selected or the problem at hand"
                             " is not a classification problem"]})
        check_results['df_conflicting_labels'] = df_conflicting_labels
        check_results['df_class_imbalance'] = df_class_imbalance
        check_results['df_feature_label_correlation'] = df_feature_label_correlation
        calculated_scores, DQ_label = calculate_dq_label.calculate_dataset_nutrition_label(df, check_results)
        missing_and_duplicates_score = round((calculated_scores['duplicate_instances'] + calculated_scores['missing_values'] + calculated_scores['duplicate_columns'])/3,2)
        Type_integrity_score = round((calculated_scores['amount_of_diff_values'] + calculated_scores['mixed_data_types'] + calculated_scores['string_mismatch'] + calculated_scores['special_characters'])/4,2)
        outliers_and_correlations_score = round((calculated_scores['feature_correlations'] + calculated_scores['outliers'])/2,2)
        label_purity_score = round(calculated_scores['conflicting_labels'],2)
        return html.Div([  # Data issue / check results section
            html.H3('Profiling report and issue overview', style={'textAlign': 'center'}),
            html.P('This section contains a profling report showing important information'
                   ' regarding ML issues found in the dataset', style={'textAlign': 'center'}),
            # TODO: TQDM loader
            # TODO: warning reports + general profiling section (o.b.v. data readiness report?)

            dbc.Table.from_dataframe(pandas_dq_report, striped=False, bordered=True, hover=True, style={
                'overflowX': 'scroll'
            }),
            html.Hr(),
            html.H3('Data quality checks', style={'textAlign': 'center'}),
            html.P('This section contains a detailed analysis of possible data quality issues in your dataset', style={'textAlign': 'center'}),
            dbc.Accordion(children=[
                              dbc.AccordionItem(
                                  [
                                      #dbc.AccordionControl("Duplicates & missing values ({})".format(36)),
                                                        html.H3('Missing values check', style={'textAlign': 'center'}),
                                                          html.P(
                                                              'Checks the type and amount of missing values. The potential total missingness column is the'
                                                              ' percent missing plus some missingness types (zeros, "?" and "-") that are often used to indicate missing values',
                                                              style={'textAlign': 'center'}),
                                                          dbc.Table.from_dataframe(df_missing_values, striped=False,
                                                                                   bordered=True, hover=True, style={
                                                                  'overflowX': 'scroll'
                                                              }),
                                                          # html.img(src=missingno_plot_src, alt="MSNO plot", width="750", height="500"),
                                                          html.Hr(),
                                                          html.H3('Duplicates check', style={'textAlign': 'center'}),
                                                          html.P(
                                                              'Checks whether there are any duplicates and displays the row numbers of the  duplicate instances.',
                                                              style={'textAlign': 'center'}),
                                                          dbc.Table.from_dataframe(df_duplicates, striped=False,
                                                                                   bordered=True, hover=True, style={
                                                                  'overflowX': 'scroll'
                                                              }),
                                                          html.H3('Duplicate columns check',
                                                                  style={'textAlign': 'center'}),
                                                          html.P(
                                                              'Checks whether there are any exact duplicate columns (which slows down the training time of your ML model).',
                                                              style={'textAlign': 'center'}),
                                                          dbc.Table.from_dataframe(df_duplicate_columns, striped=False,
                                                                                   bordered=True, hover=True, style={
                                                                  'overflowX': 'scroll'
                                                              }),
                                                          # dcc.Graph(figure=mpl_to_plotly(missingno_plot))

                                  ], title="Duplicates & missing values ({}/100)".format(missing_and_duplicates_score),
                              ),
                              dbc.AccordionItem(
                                  [
                                      #dbc.AccordionControl("Type integrity ({})".format(36)),
                                      html.H3('Amount of distinct values per column',
                                                                  style={'textAlign': 'center'}),
                                                          html.P(
                                                              'Checks the amount of different values for each column, consisting in the {} samples.'.format(
                                                                  len(df)),
                                                              style={'textAlign': 'center'}),
                                                          dbc.Table.from_dataframe(df_amount_of_diff_values,
                                                                                   striped=False, bordered=True,
                                                                                   hover=True, style={
                                                                  'overflowX': 'scroll'
                                                              }),
                                                          html.Hr(),

                                                          html.H3('Mixed data types check',
                                                                  style={'textAlign': 'center'}),
                                                          html.P(
                                                              'Checks for different data types in your dataset, and displays some random samples.',
                                                              style={'textAlign': 'center'}),
                                                          dbc.Table.from_dataframe(df_mixed_data_types, striped=False,
                                                                                   bordered=True, hover=True, style={
                                                                  'overflowX': 'scroll'
                                                              }),
                                                          html.Hr(),

                                                          html.H3("Special characters check",
                                                                  style={'textAlign': 'center'}),
                                                          html.P(
                                                              "Checks for data points that contain only special characters like '?!$^&#'.",
                                                              style={'textAlign': 'center'}),
                                                          dbc.Table.from_dataframe(df_special_characters, striped=False,
                                                                                   bordered=True, hover=True, style={
                                                                  'overflowX': 'scroll'
                                                              }),
                                                          html.Hr(),
                                                          html.H3("String mismatch / cell entity check",
                                                                  style={'textAlign': 'center'}),
                                                          html.P(
                                                              "Checks for strings that have the same base form, like 'red', 'Red', 'RED!' (base form 'red' ).",
                                                              style={'textAlign': 'center'}),
                                                          dbc.Table.from_dataframe(df_string_mismatch, striped=False,
                                                                                   bordered=True, hover=True, style={
                                                                  'overflowX': 'scroll'
                                                              })], title="Type integrity ({}/100)".format(Type_integrity_score),
                              ),
                              dbc.AccordionItem(
                                  [
                                      #dbc.AccordionControl("Outliers & correlations ({})".format(36)),
                                      html.H3("Outlier samples check",
                                                                  style={'textAlign': 'center'}),
                                                          html.P([
                                                                     "Function that checks for outliers samples (jointly across all features) using "
                                                                     "the LoOP algorithm: ", html.A('LoOp paper.',
                                                                                                    href='https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf')],
                                                                 style={'textAlign': 'center'}),
                                                          dbc.Table.from_dataframe(df_outliers, striped=False,
                                                                                   bordered=True, hover=True,
                                                                                   style={'overflowX': 'scroll'}),
                                                          html.P(
                                                              "{} outliers have been found above the set probability threshold of {}.".format(
                                                                  amount_of_outliers, threshold),
                                                              style={'textAlign': 'center'}),
                                                          html.P(
                                                              "The parallel coordinates plot below can be used to identify outliers and correlations in your dataset",
                                                              style={'textAlign': 'center'}),
                                                          dcc.Graph(figure=pcp_plot),
                                                          html.P(
                                                              "Lookup table for the label mappings.",
                                                              style={'textAlign': 'center'}),
                                                          dbc.Table.from_dataframe(label_mapping, striped=False,
                                                                                   bordered=True, hover=True, style={
                                                                  'overflowX': 'scroll'
                                                              }),
                                                          html.P(
                                                              "Potential outliers are visualized in the boxplot below.",
                                                              style={'textAlign': 'center'}),
                                                          dcc.Graph(figure=box_plot),  # TODO: checken
                                                          html.Hr(),
                                                          html.H3("Feature-feature correlation check",
                                                                  style={'textAlign': 'center'}),
                                                          html.P("computes the correlation between each feature pair;"
                                                                 " Methods to calculate for each feature label pair:"
                                                                 " numerical-numerical: Pearson’s correlation coefficient"
                                                                 " numerical-categorical: Correlation ratio"
                                                                 " categorical-categorical: Symmetric Theil’s U.",
                                                                 style={'textAlign': 'center'}),
                                                          dcc.Graph(figure=correlationFig),
                                                          html.Hr(),
                                                          html.H3("Feature-label correlation check",
                                                                  style={'textAlign': 'center'}),
                                                          html.P(
                                                              "Computes the correlation between each feature and the label, "
                                                              "in a similar fashion as the feature-feature correlation.",
                                                              style={'textAlign': 'center'}),
                                                          dbc.Table.from_dataframe(df_feature_label_correlation,
                                                                                   striped=False, bordered=True,
                                                                                   hover=True, style={
                                                                  'overflowX': 'scroll'
                                                              })], title="Outliers & correlations ({}/100)".format(outliers_and_correlations_score),
                              ),
                              dbc.AccordionItem(
                                  [
                                      #dbc.AccordionControl("Label purity ({})".format(36)),
                                          html.H3('Class imbalance check', style={'textAlign': 'center'}),
                                           html.P('Checks the distribution of instances per label.',
                                                  style={'textAlign': 'center'}),
                                           dbc.Table.from_dataframe(df_class_imbalance, striped=False, bordered=True,
                                                                    hover=True, style={
                                                   'overflowX': 'scroll'
                                               }),  # TODO: checken hoe dit zit bij regression
                                           html.Hr(),
                                           html.H3('Conflicting labels check',
                                                   style={'textAlign': 'center'}),
                                           html.P(
                                               'Checks for instances with exactly the same feature values, but different labels (which can confuse your ML model).',
                                               style={'textAlign': 'center'}),
                                           dbc.Table.from_dataframe(df_conflicting_labels, striped=False, bordered=True,
                                                                    hover=True, style={
                                                   'overflowX': 'scroll'
                                               }),
                                           html.P("There are {}% conflicting labels.".format(percent_conflicting),
                                                  style={'textAlign': 'center'}),

                                           ], title="Label purity ({}/100)".format(label_purity_score),
                                          ),
                          ], start_collapsed=True,

        ),
        html.Div(dcc.Loading(children=html.Div(id='bias_and_feature_information_accordion'), type='circle')),
        dq_checks_overview(calculated_scores, DQ_label),

        #TODO: hier progress bars & data quality label
        #progress_bars(),

        html.Hr(),  # horizontal line
        html.H3('Additional checks', style={'textAlign': 'center'}),
        html.P('Press the "Run additional checks" button to detect potential label errors using Cleanlab'
               ' & to perform a baseline performance assessment by training three ML models on your dataset.', style={'textAlign': 'center'}),
        #html.Button('Run additional checks', id='run-long-running-time-checks-button', n_clicks=0, style=button_style),
        html.Div(dbc.Button('Run additional checks', id='run-long-running-time-checks-button', n_clicks=0, color='primary'), style={"display": "flex", "justify-content": "center", "align-items": "center", "height": "100%"}),
        html.Hr(),
        html.Div(dcc.Loading(children=html.Div(id='long_running_time_accordion'), type='circle')), #TODO: boosdoener traagheid
        ])
    else:
        html.Hr()


@app.callback(
    Output('bias_and_feature_information_accordion', 'children'),
    [Input('run-checks-button', 'n_clicks'),
     State('stored-filepath', 'data'),
     State('dtypes_dropdown', 'data'),
     State('targetColumn', 'value')]
)
def bias_and_feature_information_accordion(n_clicks, filepath, dtypes, target):#, n, target):
    if n_clicks >= 1: #run checks button clicked
        dtypes_dict = dtypes[0]
        task_type = 'other'
        if target != 'None':
            if dtypes_dict[target] == 'categorical' or dtypes_dict[target] == 'boolean':
                task_type = 'classification'
            else:
                task_type = 'regression'

        df = fetch_data(filepath)

        #df = plot_and_transform_functions.clean_dataset(df)
        distribution_figures = plot_and_transform_functions.plot_dataset_distributions(df, dtypes_dict) #list of all column data distribution figures
        data_distribution_figures_div = html.Div([dcc.Graph(id='multi_' + str(i), figure=distribution_figures[i], style={'display': 'inline-block', 'width': '30vh', 'height': '30vh'}) for i in range(len(distribution_figures))])


        return dbc.Accordion(
                            children=[
                                dbc.AccordionItem(
                                    [
                                        #dbc.AccordionControl("Bias & feature information ({})".format(36)),
                                            # Data distribution plots
                                            html.H3('Data distribution', style={'textAlign': 'center'}),
                                            # data distribution plots
                                            html.P(
                                                'The following plots give insights into the datasets central tendency and spread. Each plot represents a variables distribution,'
                                                ' with the x-axis showing its value and the y-axis indicating the frequency/proportion of data points with that value.',
                                                style={'textAlign': 'center'}),

                                            data_distribution_figures_div,
                                            html.Hr(),

                                            # Feature importance plot
                                            html.H3('Feature importance analysis', style={'textAlign': 'center'}),
                                            html.P(
                                                'Displays the feature importance based on target encoded values.',
                                                style={'textAlign': 'center'}),
                                            dcc.Loading(
                                                id="loading-2",
                                                children=html.Div(id="feature_importance_plot_div")),
                                            html.Hr(),

                                            # Subgroup bias analysis
                                            html.H3('Bias analysis (DIY)', style={'textAlign': 'center'}),
                                            html.P(
                                                'In the dropdown menu below, select the sensitive features (if any) that exist in your dataset. The bar chart indicates the amount of times'
                                                ' a specific sensitive subgroup appears in your dataset, and the distribution of the class label for that subgroup'
                                                ' NOTE: it can take a while in order for the chart to be displayed.',
                                                style={'textAlign': 'center'}),
                                            dcc.Dropdown(id="biasDropdown",
                                                         options=[{'label': x, 'value': x} for x in df.columns],
                                                         multi=True,
                                                         placeholder="Select sensitive feature(s)"),
                                            dcc.Loading(
                                                id="loading-3",
                                                children=html.Div(id="biasGraph")),
                                        ], title="Bias & feature information"),
                                    ],
                                    start_collapsed=True
                                ),



@app.callback(
    Output('feature_importance_plot_div', 'children'),
    [Input('run-checks-button', 'n_clicks'),
     State('stored-filepath', 'data'),
     State('dtypes_dropdown', 'data'),
     State('targetColumn', 'value')]
)
def feature_importance_plot(n_clicks, filepath, dtypes, target):#, n, target):
    if n_clicks >= 1: #run checks button clicked

        if target != 'None':
            df = fetch_data(filepath)
            df = plot_and_transform_functions.clean_dataset(df)
            dtypes_dict = dtypes[0]
            feature_importance_plot = plot_and_transform_functions.plot_feature_importance(df, target, dtypes_dict)
            return dcc.Graph(figure = feature_importance_plot)
        else:
            return html.P('NOT APPLICABLE: The feature importance plot is not applicable when no target column is selected')


@app.callback(
    Output('biasGraph', 'children'),
    [Input('biasDropdown', 'value'),
     State('stored-filepath', 'data'),
     State('targetColumn', 'value')])

def bias_graph(bias_dropdown, filepath, target):
    if bias_dropdown:
        if target in bias_dropdown:
            bias_dropdown = bias_dropdown.remove(target) # as this just gives the class distribution, which is already presented before, and hinders logical results when
        # combined with other features
        if bias_dropdown:
            # then atleast one sensitive feature that is not the target is selected, plot the analysis
            df = fetch_data(filepath)
            df = plot_and_transform_functions.clean_dataset(df)
            if target != 'None':
                sensitive_feature_combinations_table = fairness_checks.sensitive_feature_combinations(df,bias_dropdown, target, bins=5)
                stacked_bar_chart_fig = fairness_checks.plot_stacked_barchart(sensitive_feature_combinations_table)
                return dcc.Graph(figure=stacked_bar_chart_fig)
            else:
                return html.P('NOT APPLICABLE: Bias analysis is not applicable when no target column is selected')




@app.callback(
    Output('long_running_time_accordion', 'children'),
    [Input('run-long-running-time-checks-button', 'n_clicks'),
     State('stored-filepath', 'data'),
     State('dtypes_dropdown', 'data'),
     State('targetColumn', 'value')]
)
def cleanlab_and_baseline_performance(n_clicks, filepath, dtypes, target):#, n, target):
    if n_clicks >= 1: #run checks button clicked
        if target != 'None':
            dtypes_dict = dtypes[0]
            task_type = 'other'
            if target != 'None':
                if dtypes_dict[target] == 'categorical' or dtypes_dict[target] == 'boolean':
                    task_type = 'classification'
                else:
                    task_type = 'regression'

            #label erros can only be detected when the task at hand is a classification task
            if task_type == 'classification':
                df = fetch_data(filepath)

                df = plot_and_transform_functions.clean_dataset(df)
                df, error_message = plot_and_transform_functions.upsample_minority_classes(df, target) #upsample instances that occur less than 5 times
                encoded_dataframe, mapping_encoding = plot_and_transform_functions.encode_categorical_columns(df, target,
                                                                                                              dtypes_dict)
                _, issues_dataframe_only_errors, wrong_label_count, accuracy_model = label_purity.cleanlab_label_error(encoded_dataframe, target)
                issues_dataframe_only_errors['given_label'] = issues_dataframe_only_errors['given_label'].astype(int)
                issues_dataframe_only_errors['given_label'] = issues_dataframe_only_errors['given_label'].map(mapping_encoding)
                issues_dataframe_only_errors['predicted_label'] = issues_dataframe_only_errors['predicted_label'].astype(int)
                issues_dataframe_only_errors['predicted_label'] = issues_dataframe_only_errors['predicted_label'].map(
                    mapping_encoding)
            else:
                issues_dataframe_only_errors = pd.DataFrame(
                {"Check notification": ["This check is not applicable as there is no target column selected or the problem at hand"
                             " is not a classification problem"]})
                wrong_label_count = 0
                accuracy_model = 0

            return dbc.Accordion(
                                children=[
                                    dbc.AccordionItem(
                                        [
                                            # dmc.AccordionControl("Model performance & label errors ({})".format(36)),

                                                                html.H3('Cleanlab.ai label issues check', style={'textAlign': 'center'}),
                                                                html.P([
                                                                    "Automatically detects probable label errors using Confident learning, ",
                                                                    html.A('Confident learning paper',
                                                                           href='https://arxiv.org/abs/1911.00068'),
                                                                    ' . The numeric label quality score quantifies clenlabs confidence that the label is correct '
                                                                    '(a low score thus indicates a high probability that the label is wrong). The accuracy of the XgBoost Classifier model used to make the predictions'
                                                                    ' is {}%, when this score is low, cleanlabs predictions could be less accurate. {}'.format(accuracy_model, error_message)],
                                                                    style={'textAlign': 'center'}),
                                                                # html.Div(dcc.Loading(id='loading-4',children=html.Div(id="cleanlab_table"))),
                                                                html.H6(f'Cleanlab detected {wrong_label_count} potential label errors.'),
                                                                dash_table.DataTable(issues_dataframe_only_errors.to_dict('records'),
                                                                        page_size = 10,
                                                                        editable=False,
                                                                        style_table={
                                                                            'overflowX': 'scroll'
                                                                        }),
                                                                html.Hr(),
                                                                #Simple model performance
                                                                html.H3('Baseline Performance Assessment', style={'textAlign': 'center'}),
                                                                html.P('Displays the baseline performance of three sklearn models with basic settings.'
                                                                       ' Note that this is in general definitely not the best performance you can achieve,'
                                                                       ' always tune your model parameters (e.g. by performing a gridsearch) to increase model performance.',
                                                                       style={'textAlign': 'center'}),
                                                                dcc.Loading(
                                                                    id="loading-5",
                                                                    children=html.Div(id="baseline_performance_plot")),
                                                                # dcc.Graph(figure=mpl_to_plotly(missingno_plot))
                                                        ], title="Label errors & baseline performance",
                                                        )])


@app.callback(
    Output('baseline_performance_plot', 'children'),
    [Input('run-checks-button', 'n_clicks'),
     State('stored-filepath', 'data'),
     State('dtypes_dropdown', 'data'),
     State('targetColumn', 'value')]
)
def baseline_performance_assessment(n_clicks, filepath, dtypes, target):
    if n_clicks >= 1: #run checks button clicked

        if target != 'None':
            df = fetch_data(filepath)
            df = plot_and_transform_functions.clean_dataset(df)
            dtypes_dict = dtypes[0]
            encoded_dataframe, mapping_encoding = plot_and_transform_functions.encode_categorical_columns(df, target,
                                                                                                          dtypes_dict)
            baseline_performance_plot = plot_and_transform_functions.baseline_model_performance(encoded_dataframe, target, dtypes_dict)
            return dcc.Graph(figure = baseline_performance_plot)


def generate_filepath(uploaded_filename):
    session_id = str(uuid.uuid4())
    uploaded_filename = os.path.splitext(uploaded_filename)[0] #remove the file extension
    filename = f"{session_id}_{uploaded_filename}.pkl"
    filepath = os.path.join(cache_dir, filename)
    return filepath

def fetch_data(filepath):
    if os.path.exists(filepath):
        # If the file already exists, load the DataFrame from the cache
        df = pd.read_pickle(filepath)
        return df


def dq_checks_overview(check_results, DQ_label):

    if DQ_label == 'A':
        DQ_button_color = 'success'
    elif DQ_label == 'B' or DQ_label == 'C':
        DQ_button_color = 'warning'
    else:
        DQ_button_color = 'danger'

    return html.Div(
        [html.Hr(),
         html.H3('Data quality check results overview & DQ label', style={'textAlign': 'center'}),
         html.P('The bars underneath give a short summary about your check results. Green means that the check is passed, red means that the check is failed,'
                ' and yellow indicates a warning. The yellow scores are not taken into account in the calculation of the  data quality label.', style={'textAlign': 'center'}),
         dbc.Row([
            dbc.Col(dbc.Progress(value=check_results["missing_values"], label=f"{check_results['missing_values']}%",
                         color=f"{check_results['missing_values_color']}", className="mb-3")),
            dbc.Col(html.H6("Missing values check (fails when one or more columns exceed 5% missingness)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["duplicate_instances"], label=f"{check_results['duplicate_instances']}%",
                                  color=f"{check_results['duplicate_instances_color']}", className="mb-3")),
             dbc.Col(html.H6("Duplicate instances check (fails when the percentage of duplicate data exceeds 1%.)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["duplicate_columns"], label=f"{check_results['duplicate_columns']}%",
                                  color=f"{check_results['duplicate_columns_color']}", className="mb-3")),
             dbc.Col(html.H6("Duplicate columns check (fails when one or more duplicate column has been found)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["mixed_data_types"], label=f"{check_results['mixed_data_types']}%",
                                  color=f"{check_results['mixed_data_types_color']}", className="mb-3")),
             dbc.Col(html.H6("Mixed data types check (fails when one or more columns contain mixed data types)"))]),

         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["string_mismatch"], label=f"{check_results['string_mismatch']}%",
                                  color=f"{check_results['string_mismatch_color']}", className="mb-3")),
             dbc.Col(html.H6("String mismatch check (fails when one or more data points have been found with multiple variant of the same base form)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["outliers"], label=f"{check_results['outliers']}%",
                                  color=f"{check_results['outliers_color']}", className="mb-3")),
             dbc.Col(html.H6("Outliers check (fails when the percentage of outliers exceeds 1%)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["conflicting_labels"], label=f"{check_results['conflicting_labels']}%",
                                  color=f"{check_results['conflicting_labels_color']}", className="mb-3")),
             dbc.Col(html.H6("Conflicting labels check (fails when one or more conflicting labels have been found)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["amount_of_diff_values"], label=f"{check_results['amount_of_diff_values']}%",
                                  color=f"{check_results['amount_of_diff_values_color']}", className="mb-3")),
             dbc.Col(html.H6("Single value check (fails when one or more columns exist that contain a single value)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["special_characters"],
                                  label=f"{check_results['special_characters']}%",
                                  color=f"{check_results['special_characters_color']}", className="mb-3")),
             dbc.Col(html.H6("Special characters check (fails when one or more columns exist that contain a data point consisting of only special characters)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["feature_correlations"],
                                  label=f"{check_results['feature_correlations']}%",
                                  color=f"{check_results['feature_correlations_color']}", className="mb-3")),
             dbc.Col(html.H6("Feature-feature correlation check (fails when one or more column pairs exist with a correlation >0.9 or <-0.9)"))]),

         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["feature_label_correlation"],
                                  label=f"{check_results['feature_label_correlation']}%",
                                  color=f"{check_results['feature_label_correlation_color']}", className="mb-3")),
             dbc.Col(html.H6("Feature label correlation check (gives a warning when a feature is correlated >0.9 with the target)"))]),

         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["class_imbalance"],
                                  label=f"{check_results['class_imbalance']}%",
                                  color=f"{check_results['class_imbalance_color']}", className="mb-3")),
             dbc.Col(html.H6("Class imbalance check (gives a warning when the ratio of the rarest and the most common classe is high)"))]),

            html.Div(dbc.Button(f'DQ label: {DQ_label}', color=DQ_button_color, className="mr-3"), style={"display": "flex", "justify-content": "center", "align-items": "center", "height": "100%"}),
            html.Hr()
        ]
    )

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)
