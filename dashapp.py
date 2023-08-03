import pandas as pd
import deepchecks
import dash
from dash import html, dcc, dash_table
import io
import base64
from dash.dependencies import Input, Output, State
import uuid
import scipy.io.arff as arff
import calculate_dq_label
from DataTypeInference import obtain_feature_type_table, createDatasetObject
import fairness_checks
import Duplicates_and_Missing
import Type_integrity
import outliers_and_correlations
import dash_bootstrap_components as dbc
import label_purity
import plot_and_transform_functions
import dq_improvements
import os
import nltk
import logging
from dash.exceptions import PreventUpdate
#silence unneccesary warnings

deepchecks.set_verbosity(logging.ERROR)
os.environ['DISABLE_LATEST_VERSION_CHECK'] = 'True'

#list all dtype options for drop down menu
sortingHatInf_datatypes = ['not-generalizable', 'categorical', 'boolean', 'datetime', 'sentence', 'url', 'embedded-number', 'list', 'context-specific', 'numeric']

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css" #dbc stylesheet
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc_css]

button_style = {'background-color': 'blue',
                    'color': 'white',
                    'height': '50px',
                    'margin-top': '50px',
                    'margin-left': '50px'}

try: #somehow nltk does not by default install the stopwords module, so do it if users do not have it
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
current_directory = os.path.dirname(os.path.realpath('dashapp.py'))
cache_dir = os.path.join(current_directory, 'cached_files')



app.title = "Data Quality Analyzer"
#app layout
app.layout = dbc.Container(html.Div([
    html.H1("Data Quality Analyzer", style={'fontSize':50, 'textAlign':'center'}),
    html.Hr(),
    dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ], style={
            'fontFamily': 'Arial',
                }),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'solid',  # Solid border
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'background': '#007BFF',  # Blue background
            'color': 'white',  # White text color
            'fontFamily': 'Arial',  # Specify the font
            },
            max_size=-1,
            multiple=True
        ),
     #first screen with instructions on how to use the tool
     dcc.Loading(children=html.Div(id='output-data-upload'), type = 'circle', style={'content': "Loading..."}),


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
        return dbc.Container(html.Div([html.Br(),html.H5('Welcome to the data quality toolkit for assessing, analyzing tabular & improving machine learning datasets', style={'textAlign': 'left'}),
                                       html.Ol(
                                        children=[
                                            html.Li('Upload a file (allowed file types: .csv, .xls, .xlsx, .arff, .parquet).'),
                                            html.Li('Select your target column, and if necessary, correct the automatically inferred data types of your columns.'),
                                            html.Li("Press 'Run checks', your checks will now be run. (change preferences in Advanced settings)."),
                                            html.Li("Optional (task dependent): Press 'Run additional checks' to find label errors in your dataset and see the baseline performance on"
                                                    " three ML models. In addition, you can perform a bias analysis."),
                                            html.Li(
                                                "Accept & apply the recommended issue remediations and download the improved dataset with just a few clicks!")

                                        ], style={'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': '15px'}
                                    )]), fluid=True, class_name="dbc")

def parse_contents(contents, filename, date):
    """"decode uploaded files depending on the specific format uploaded"""
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

        #save df locally to enable storage of larger datasets
        df.to_pickle(filepath)
        featureTypeTable = obtain_feature_type_table(df) #sortinghatinf dtype inference
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file. Make sure to upload one of the following file types (.csv, .xls, .xlsx, .arff, .parquet)',
            html.Br(),
            dbc.Button('DQ label: E', color='danger') #assign bad data readiness label if we can't read the file
        ], className="d-grid gap-2 col-6 mx-auto")

    return dbc.Container(html.Div([
        html.P('Uploaded file: {}'.format(filename)),
        html.H6('Choose your target column'),
        dcc.Dropdown(id="targetColumn", options=[{'label':'None', 'value':'None'}] + [{'label':x, 'value':x} for x in df.columns], value = 'None'),
        html.Div(id='target-selected-container'),
        dcc.Store(id='stored-filepath', data=filepath, storage_type='memory'), #save to obtain df later on

        #Feature type table (adjustable)
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
        ),
        #let user set customized thresholds
        advanced_settings_accordion(),

        #where the magic happens
        dbc.Button('Run checks', id='run-checks-button', n_clicks=0, color='primary'),
        html.Hr(),  # horizontal line
        #Main container including all check results
        html.Div(dcc.Loading(children=html.Div(id='container-checks-button-pressed'), type='cube')),


        #HITL remediations container
        html.Div(id='cleaning-assistant-container'),
    ]), fluid = True, class_name="dbc")
@app.callback(
    Output('target-selected-container', 'children'),
    Input('targetColumn', 'value')#, Input('submit-button','n_clicks'), State('targetColumn','targetvalue')]
)
def display_selected_target_column(value):#, n, target):
    return f'You have selected {value}, as your target column'

@app.callback(
    Output('container-checks-button-pressed', 'children'),
    # Output('cleaning-assistant-container', 'children'),
    [Input('run-checks-button', 'n_clicks'),
     State('stored-filepath', 'data'),
     State('dtypes_dropdown', 'data'),
     State('targetColumn', 'value'),
    #Advanced settings
    State('advanced_settings_missing', 'value'),
    State('advanced_settings_duplicates', 'value'),
    State('advanced_settings_outliers', 'value'),
    State('advanced_settings_correlation', 'value'),
    State('advanced_settings_duplicate_columns', 'value'),
    State('advanced_settings_single_value', 'value'),
    State('advanced_settings_mixed_data_types', 'value'),
    State('advanced_settings_special_characters', 'value'),
    State('advanced_settings_string_mismatch', 'value'),
    State('advanced_settings_conflicting_labels', 'value'),
    State('advanced_settings_10k', 'value')]
)
def run_checks(n_clicks, filepath, dtypes, target, missing, duplicates, outliers, correlation, duplicate_columns, single_value,
                  mixed_data_types, special_characters, string_mismatch, conflicting_labels, sample_10k):#, n, target):
    """"Power train of the code, here all checks are run and visualized in the desired format"""

    if n_clicks >= 1: #run checks button clicked
        dtypes_dict = dtypes[0]
        task_type = 'other'
        if target != 'None': #check task type
            if dtypes_dict[target] == 'categorical' or dtypes_dict[target] == 'boolean':
                task_type = 'classification'
            else:
                task_type = 'regression'
        #if user manually set thresholds, make sure to use them
        settings_dict = {
            'advanced_settings_missing': missing,
            'advanced_settings_duplicates': duplicates,
            'advanced_settings_outliers': outliers,
            'advanced_settings_correlation': correlation,
            'advanced_settings_duplicate_columns': duplicate_columns,
            'advanced_settings_single_value': single_value,
            'advanced_settings_mixed_data_types': mixed_data_types,
            'advanced_settings_special_characters': special_characters,
            'advanced_settings_string_mismatch': string_mismatch,
            'advanced_settings_conflicting_labels': conflicting_labels,
            'advanced_settings_10k': sample_10k
        }

        result_strings = []


        df = fetch_data(filepath) #obtain locally stored data (to prevent errors from storing large datasets in browser)

        #Sample 10000 rows if user uploaded a large dataframe, unless specified otherwise in the advanced settings
        if len(df) > 10000 and settings_dict['advanced_settings_10k'] == True:
            df = df.sample(n=10000, random_state=42)
            df.to_pickle(filepath) #save smaller df in place of the previous df
        ds = createDatasetObject(df, dtypes_dict, target) #create deepchecks dataset object (as it is more complete with target specification)

        #RUNNING OF THE CHECKS
        check_results = {} #dictionary for saving check results
        rows, cols = df.shape

        #duplicates & missing (checks 1 up until 4)
        df_missing_values = Duplicates_and_Missing.missing_values(df)
        missing_value_mask = df.isna()
        total_missing_values = missing_value_mask.sum().sum()

        total_missing_percentage = (total_missing_values/(rows*cols))*100
        check_results['df_missing_values'] = total_missing_percentage
        df_duplicates, duplicates_result_string = Duplicates_and_Missing.duplicates(df, dtypes_dict)
        result_strings.append(duplicates_result_string)
        check_results['df_duplicates'] = df_duplicates

        df_duplicate_columns, duplicate_columns_result_string = Duplicates_and_Missing.duplicate_column(df)
        result_strings.append(duplicate_columns_result_string)
        check_results['df_duplicate_columns'] = df_duplicate_columns

        #type integrity checks (checks 8 up until 11)
        df_amount_of_diff_values = Type_integrity.amount_of_diff_values(df)

        check_results['df_amount_of_diff_values'] = df_amount_of_diff_values
        df_mixed_data_types = Type_integrity.mixed_data_types(df)
        check_results['df_mixed_data_types'] = df_mixed_data_types

        first_row_numeric = pd.to_numeric(df_mixed_data_types.iloc[0], errors='coerce')  # convert strings to numeric
        mixed_columns_test = (first_row_numeric > 0) & (first_row_numeric < 1)
        mixed_dtypes_dict = mixed_columns_test.to_dict() #necessary for encoding later on
        df_special_characters = Type_integrity.special_characters(df)

        check_results['df_special_characters'] = df_special_characters

        df_string_mismatch = Type_integrity.string_mismatch(df)
        check_results['df_string_mismatch'] = df_string_mismatch

        #outliers & correlations (check 14&15)
        df_feature_feature_correlation, correlationFig = outliers_and_correlations.feature_feature_correlation(ds)

        check_results['df_feature_feature_correlation'] = df_feature_feature_correlation
        df_outliers, amount_of_outliers, threshold, outlier_prob_scores, outlier_result_string = outliers_and_correlations.outlier_detection(ds)
        result_strings.append(outlier_result_string)
        check_results['df_outliers'] = df_outliers

        #Extended pandas_dq report; extended to contain more checks and existing checks presented more complete
        pandas_dq_report_adjusted = plot_and_transform_functions.pandas_dq_report(df, dtypes_dict, df_mixed_data_types, df_special_characters,
                                                                                  df_string_mismatch, df_feature_feature_correlation ,target)

        #remove columns with huge missingness
        df_cleaned = plot_and_transform_functions.clean_dataset(df)

        #Parallel coordinate plot color coded on outlier probability values to verify outliers
        label_encoded_df, label_mapping = plot_and_transform_functions.label_encode_dataframe(df_cleaned, dtypes_dict) #mapping numbers --> original data
        pcp_plot = plot_and_transform_functions.pcp_plot(label_encoded_df, target, outlier_prob_scores)
        box_plot = outliers_and_correlations.box_plot(df_cleaned, dtypes_dict) #for verifying numerical column outliers

        if task_type == 'classification': #target column supplied
            df_feature_label_correlation = outliers_and_correlations.feature_label_correlation(ds)
            #label purity checks (check 5&6)
            df_class_imbalance, fig_class_imbalance, class_imbalance_result_string = label_purity.class_imbalance(ds)
            result_strings.append(class_imbalance_result_string)
            df_conflicting_labels, percent_conflicting, conflicting_labels_result_string = label_purity.conflicting_labels(ds)
            result_strings.append(conflicting_labels_result_string)
        else: #no target column selected or not a classification problem, thus not applicable; create placeholders
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
        #store results in dict
        check_results['df_conflicting_labels'] = df_conflicting_labels
        check_results['df_class_imbalance'] = df_class_imbalance
        check_results['df_feature_label_correlation'] = df_feature_label_correlation

        #determine which checks are passed and calculate final data readiness level
        calculated_scores, DQ_label, executive_summary = calculate_dq_label.calculate_dataset_nutrition_label(df, check_results, settings_dict)
        #obtain problematic indices for data cleaning later on

        dq_issues_dict = dq_improvements.obtain_indices_with_issues(df, check_results, settings_dict)
        #add ID columns
        for col in df.columns:
            if dtypes_dict[col] == 'not-generalizable':
                if col not in dq_issues_dict['redundant_columns']:
                    dq_issues_dict['redundant_columns'].append(col)


        #scores for displaying in accordion
        missing_and_duplicates_score = round((calculated_scores['duplicate_instances'] + calculated_scores['missing_values'] + calculated_scores['duplicate_columns'])/3,2)
        Type_integrity_score = round((calculated_scores['amount_of_diff_values'] + calculated_scores['mixed_data_types'] + calculated_scores['string_mismatch'] + calculated_scores['special_characters'])/4,2)
        outliers_and_correlations_score = round((calculated_scores['feature_correlations'] + calculated_scores['outliers'])/2,2)
        label_purity_score = round(calculated_scores['conflicting_labels'],2)
        #column independent information for underneath the extended pandas_dq report
        non_empty_strings = [s for s in result_strings if s.strip()] # Filter out the empty strings
        result_string = '  \n'.join(non_empty_strings)
        result_string = result_string.upper()

        return html.Div([  # Data issue / check results section
            dbc.Row(
                [
                    dbc.Col(dbc.Button("1", color="secondary", outline=True, className="font-weight-bold",
                                       style={'fontSize': '20px', 'padding': '10px', 'width': '50px'})),
                    dbc.Col([
                        html.H3('Profiling report and issue overview', style={'textAlign': 'center', "margin-right": "50px"}),
                        html.P('This section contains a profling report showing important information,'
                               ' regarding ML issues found in the dataset. In addition, the executive summary of your dataset is displayed.',
                               style={'textAlign': 'center', "margin-right": "50px"}),
                    ], width=11, align='center'),
                ],
                align="center", justify="center"
            ),
            dbc.Container([
                dbc.Alert([
                    html.H4("Executive summary", className="alert-heading", style={'textAlign': 'center'}),
                    executive_summary
                ], color="secondary")
            ]),
            dbc.Table.from_dataframe(pandas_dq_report_adjusted, striped=False, bordered=True, hover=True, style={
                'overflowX': 'scroll'
            }),
            dcc.Markdown(result_string),
            html.Hr(style={'borderWidth': '10px', 'borderColor': 'black'}),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("2", color="secondary", outline=True, className="font-weight-bold",
                                       style={'fontSize': '20px', 'padding': '10px', 'width': '50px'})),
                    dbc.Col([
                        html.H3('Data quality checks', style={'textAlign': 'center', "margin-right": "250px"}),
                        html.P('This section contains a detailed analysis of possible data quality issues in your dataset.',
                               style={'textAlign': 'center', "margin-right": "250px"}),
                    ], width=10, align='center'),
                ],
                justify="around"
            ),
            dbc.Accordion(children=[
                              dbc.AccordionItem(
                                  [
                                    html.H3('Missing values check', style={'textAlign': 'center'}),
                                      html.P(
                                          'Checks the type and amount of missing values. The potential total missingness column is the'
                                          ' percent missing plus some missingness types (zeros, "?" and "-") that are often used to indicate missing values',
                                          style={'textAlign': 'center'}),
                                      dbc.Table.from_dataframe(df_missing_values, striped=False,
                                                               bordered=True, hover=True, style={
                                              'overflowX': 'scroll'
                                          }),
                                      html.P("The dataset contains {}% missing instances.".format(round(total_missing_percentage,2))),

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

                                  ], title="Duplicates & missing values ({}/100)".format(missing_and_duplicates_score),
                              ),
                              dbc.AccordionItem(
                                  [
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
                                          "{} outliers have been found with a probability higher than {}.".format(
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
                                          "Potential column value outliers are visualized in the boxplot below.",
                                          style={'textAlign': 'center'}),
                                      dcc.Graph(figure=box_plot),  # TODO: checken
                                      html.Hr(),
                                      html.H3("Feature-feature correlation check",
                                              style={'textAlign': 'center'}),
                                      html.P("computes the correlation between each feature pair;"
                                             " Methods to calculate for each feature label pair:"
                                             " numerical-numerical: Pearson’s correlation coefficient;"
                                             " numerical-categorical: Correlation ratio;"
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
                                   html.P("There are {}% conflicting labels.".format(round(percent_conflicting,2)),
                                          style={'textAlign': 'center'}),

                                   ], title="Label purity ({}/100)".format(label_purity_score),
                                          ),
                          ], start_collapsed=True,

        ),
        #section for checks 16, 17 & 18
        html.Div(dcc.Loading(children=html.Div(id='bias_and_feature_information_accordion'), type='circle')),
        dq_checks_overview(calculated_scores, DQ_label, settings_dict, executive_summary),
        dcc.Store(id='mixed_dtypes_storage', data=mixed_dtypes_dict, storage_type='memory'),  # save to obtain df later on
        html.Hr(style={'borderWidth': '10px', 'borderColor': 'black'}),  # horizontal line
        dbc.Row(
            [
                dbc.Col(dbc.Col(dbc.Button("4", color="secondary", outline=True, className="font-weight-bold",
                                       style={'fontSize': '20px', 'padding': '10px', 'width': '50px'}))),
                dbc.Col([
                    html.H3('Additional checks', style={'textAlign': 'center', "margin-right": "300px"}),
                    html.P(
                        'Press the "Run additional checks" button to detect potential label errors using Cleanlab'
               ' & to perform a baseline performance assessment by training three ML models on your dataset. Note that these check may take a while to run.',
                        style={'textAlign': 'center', "margin-right": "300px"}),
                ], width=10, align='center'),
            ],
            align="center", justify="center"
        ),
        #section for check 7 & 19
        html.Div(dbc.Button('Run additional checks', id='run-long-running-time-checks-button', n_clicks=0, color='primary'), style={"display": "flex", "justify-content": "center", "align-items": "center", "height": "100%", "margin-left": "10px"}),
        html.Hr(style={'borderWidth': '10px', 'borderColor': 'black'}),
        html.Div(dcc.Loading(children=html.Div(id='long_running_time_accordion'), type='circle')),
            dcc.Store(id='store-dq_issues_dict', data=dq_issues_dict),
        layout_HITL_remediations(filepath, df_string_mismatch, df_special_characters, dtypes_dict, dq_issues_dict), #TODO: updaten voor dq_issues_dict
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
    """"data distribution, bias analysis and feature importance"""
    if n_clicks >= 1: #run checks button clicked
        dtypes_dict = dtypes[0]
        task_type = 'other'
        if target != 'None':
            if dtypes_dict[target] == 'categorical' or dtypes_dict[target] == 'boolean':
                task_type = 'classification'
            else:
                task_type = 'regression'

        df = fetch_data(filepath)
        distribution_figures = plot_and_transform_functions.plot_dataset_distributions(df, dtypes_dict) #list of all column data distribution figures
        data_distribution_figures_div = html.Div([dcc.Graph(id='multi_' + str(i), figure=distribution_figures[i], style={'display': 'inline-block', 'width': '30vh', 'height': '30vh'}) for i in range(len(distribution_figures))])
        return dbc.Accordion(
                            children=[
                                dbc.AccordionItem(
                                    [
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
                                        ], title="Feature information"),
                                dbc.AccordionItem(
                                    [
                                        #subgroup bias analysis
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
                                    ], title="Bias analysis"),
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
def feature_importance_plot(n_clicks, filepath, dtypes, target):
    """"Plot feature importance (based on target encoded values) when target variable is supplied"""
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
    """"DIY bias analysis functionality. The user selects sensitive features using a dropdown menu, and can see
    all subgroup in relation to the target feature"""
    if bias_dropdown:
        if target in bias_dropdown:
            bias_dropdown = bias_dropdown.remove(target) # as this just gives the class distribution, which is already presented before,
            # and hinders logical results when
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
     State('targetColumn', 'value'),
     State('mixed_dtypes_storage', 'data')]
)
def cleanlab_and_baseline_performance(n_clicks, filepath, dtypes, target, mixed_dtypes_dict):
    """"Check 7 & 19: label errors & baseline performance. Decoupled from the rest of the checks due to computational complexity"""
    if n_clicks >= 1: #run checks button clicked
        if target != 'None':
            dtypes_dict = dtypes[0]
            task_type = 'other'
            if target != 'None':
                #retrieve task type
                if dtypes_dict[target] == 'categorical' or dtypes_dict[target] == 'boolean':
                    task_type = 'classification'
                else:
                    task_type = 'regression'

            #label erros can only be detected when the task at hand is a classification task
            if task_type == 'classification':
                df = fetch_data(filepath)

                df = plot_and_transform_functions.clean_dataset(df)
                df, error_message = plot_and_transform_functions.upsample_minority_classes(df, target) #upsample instances that occur less than 5 times

                #encode dataframe as required by cleanlab and save mapping to convert values back later on
                encoded_dataframe, mapping_encoding = plot_and_transform_functions.encode_categorical_columns(df, target,
                                                                                                              dtypes_dict, mixed_dtypes_dict)
                _, issues_dataframe_only_errors, wrong_label_count, accuracy_model = label_purity.cleanlab_label_error(encoded_dataframe, target)
                issues_dataframe_only_errors['given_label'] = issues_dataframe_only_errors['given_label'].astype(int)
                #convert numbers displayed by cleanlab back to original data convention for readability (e.g 0 --> iris-setosa)
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
                error_message = 'No classification problem.'
            return dbc.Accordion(
                                children=[
                                    dbc.AccordionItem(
                                        [
                                            html.H3('Cleanlab.ai label issues check', style={'textAlign': 'center'}),
                                            html.P([
                                                "Automatically detects probable label errors using Confident learning, ",
                                                html.A('Confident learning paper',
                                                       href='https://arxiv.org/abs/1911.00068'),
                                                ' . The numeric label quality score quantifies clenlabs confidence that the label is correct '
                                                '(a low score thus indicates a high probability that the label is wrong). The accuracy of the XgBoost Classifier model used to make the predictions'
                                                ' is {}%, when this score is low, cleanlabs predictions could be less accurate. {}'.format(accuracy_model, error_message)],
                                                style={'textAlign': 'center'}),
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
                                    ], title="Label errors & baseline performance",
                                    )])


@app.callback(
    Output('baseline_performance_plot', 'children'),
    [Input('run-checks-button', 'n_clicks'),
     State('stored-filepath', 'data'),
     State('dtypes_dropdown', 'data'),
     State('targetColumn', 'value'),
     State('mixed_dtypes_storage', 'data')]
)
def baseline_performance_assessment(n_clicks, filepath, dtypes, target, mixed_dtypes_dict):
    """"plot baseline performance of three basic Sklearn models"""
    if n_clicks >= 1: #run checks button clicked

        if target != 'None': #nothing needs to be trained when there is no classification/regression
            df = fetch_data(filepath)
            df = plot_and_transform_functions.clean_dataset(df)
            dtypes_dict = dtypes[0]
            #encode df as numericl values are required for the models
            encoded_dataframe, mapping_encoding = plot_and_transform_functions.encode_categorical_columns(df, target,
                                                                                                          dtypes_dict, mixed_dtypes_dict)
            baseline_performance_plot = plot_and_transform_functions.baseline_model_performance(encoded_dataframe, target, dtypes_dict)
            return dcc.Graph(figure = baseline_performance_plot)


def generate_filepath(uploaded_filename):
    """"helper function for obtaining the filepath for storing the dataset offline (to prevent issues with large dataset storage in-browser)"""
    session_id = str(uuid.uuid4())
    uploaded_filename = os.path.splitext(uploaded_filename)[0] #remove the file extension
    filename = f"{session_id}_{uploaded_filename}.pkl"
    filepath = os.path.join(cache_dir, filename)
    return filepath

def fetch_data(filepath):
    """"helper function to obtain the data"""
    if os.path.exists(filepath):
        # If the file already exists, load the DataFrame from the cache
        df = pd.read_pickle(filepath)
        return df


def dq_checks_overview(check_results, DQ_label, settings_dict, executive_summary):
    """"Check result summarizing overview and data readiness label display"""
    if DQ_label == 'A':
        DQ_button_color = 'success'
    elif DQ_label == 'B' or DQ_label == 'C':
        DQ_button_color = 'warning'
    else:
        DQ_button_color = 'danger'

    return html.Div(
        [html.Hr(style={'borderWidth': '10px', 'borderColor': 'black'}),
         dbc.Row(
             [
                 dbc.Col(dbc.Button("3", color="secondary", outline=True, className="font-weight-bold",
                                    style={'fontSize': '20px', 'padding': '10px', 'width': '50px'})),
                 dbc.Col([
                     html.H3('Data quality check results overview & DQ label', style={'textAlign': 'center', "margin-right": "150px"}),
                     html.P(
                         'The bars display the degree in which the checks are passed/failed. Green means check is passed, red means check is failed,'
                ' and yellow indicates a warning. The yellow scores are not taken into account in the calculation of the data quality label.',
                         style={'textAlign': 'center', "margin-right": "150px"}),
                 ], width=10, align='center'),
             ],
             justify="around"
         ),
         dbc.Row([
            dbc.Col(dbc.Progress(value=check_results["missing_values"], label=f"{check_results['missing_values']}%",
                         color=f"{check_results['missing_values_color']}", className="mb-3")),
            dbc.Col(html.H6("Missing values check (fails when the total missingness in the dataset exceeds {}%.)".format(settings_dict['advanced_settings_missing'])))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["duplicate_instances"], label=f"{check_results['duplicate_instances']}%",
                                  color=f"{check_results['duplicate_instances_color']}", className="mb-3")),
             dbc.Col(html.H6("Duplicate instances check (fails when the percentage of duplicate data exceeds {}%.)".format(settings_dict['advanced_settings_duplicates'])))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["duplicate_columns"], label=f"{check_results['duplicate_columns']}%",
                                  color=f"{check_results['duplicate_columns_color']}", className="mb-3")),
             dbc.Col(html.H6("Duplicate columns check (fails when one or more duplicate column has been found and this was not allowed in the advanced settings)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["mixed_data_types"], label=f"{check_results['mixed_data_types']}%",
                                  color=f"{check_results['mixed_data_types_color']}", className="mb-3")),
             dbc.Col(html.H6("Mixed data types check (fails when one or more columns contain mixed data types and this was not allowed in the advanced settings)"))]),

         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["string_mismatch"], label=f"{check_results['string_mismatch']}%",
                                  color=f"{check_results['string_mismatch_color']}", className="mb-3")),
             dbc.Col(html.H6("String mismatch check (fails when one or more data points have been found with multiple variant of the same base form and this was not allowed in the advanced settings)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["outliers"], label=f"{check_results['outliers']}%",
                                  color=f"{check_results['outliers_color']}", className="mb-3")),
             dbc.Col(html.H6("Outliers check (fails when the percentage of outliers exceeds {}%)".format(settings_dict['advanced_settings_outliers'])))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["conflicting_labels"], label=f"{check_results['conflicting_labels']}%",
                                  color=f"{check_results['conflicting_labels_color']}", className="mb-3")),
             dbc.Col(html.H6("Conflicting labels check (fails when conflicting labels have been found and this was not allowed in the advanced settings)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["amount_of_diff_values"], label=f"{check_results['amount_of_diff_values']}%",
                                  color=f"{check_results['amount_of_diff_values_color']}", className="mb-3")),
             dbc.Col(html.H6("Single value check (fails when one or more columns exist that contain a single value and this was not allowed in the advanced settings)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["special_characters"],
                                  label=f"{check_results['special_characters']}%",
                                  color=f"{check_results['special_characters_color']}", className="mb-3")),
             dbc.Col(html.H6("Special characters check (fails when one or more columns exist that contain a data point consisting of only special characters and this was not allowed in the advanced settings)"))]),
         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["feature_correlations"],
                                  label=f"{check_results['feature_correlations']}%",
                                  color=f"{check_results['feature_correlations_color']}", className="mb-3")),
             dbc.Col(html.H6("Feature-feature correlation check (fails when one or more column pairs exist with a correlation >{} or <-{})".format(settings_dict['advanced_settings_correlation'],settings_dict['advanced_settings_correlation'])))]),

         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["feature_label_correlation"],
                                  label=f"{check_results['feature_label_correlation']}%",
                                  color=f"{check_results['feature_label_correlation_color']}", className="mb-3")),
             dbc.Col(html.H6("Feature label correlation check (gives a warning when a feature is correlated >{} with the target)".format(settings_dict['advanced_settings_correlation'])))]),

         dbc.Row([
             dbc.Col(dbc.Progress(value=check_results["class_imbalance"],
                                  label=f"{check_results['class_imbalance']}%",
                                  color=f"{check_results['class_imbalance_color']}", className="mb-3")),
             dbc.Col(html.H6("Class imbalance check (gives a warning when the ratio of the rarest and the most common class is high (ratio <0.1))"))]),

            html.Div([dbc.Button(f'DQ label: {DQ_label}', id="dq_label_button", color=DQ_button_color, className="mr-3"), dbc.Tooltip("{}".format(executive_summary), target="dq_label_button")],
                     style={"display": "flex", "justify-content": "center", "align-items": "center", "height": "100%"}
                     ),
            html.Hr()
        ]
    )

def advanced_settings_accordion():
    """"Contains the input & check boxes for setting adjustable thresholds for the checks
    (assuming that for some users certain issues are not an issue)"""

    return  dbc.Accordion(
        [
            dbc.AccordionItem(
                title="Advanced settings",
                children=[
                        dbc.Row([#missingess threshold
                        dbc.Col(dbc.Input(
                        id='advanced_settings_missing',  type='number',  value=5.0, min=0, max=100, ), width=2
                        ),
                        dbc.Col(html.H6("(%) Allowed missingness threshold [0,100]%"), align='center')]),

                        dbc.Row([#duplicate instances threshold
                        dbc.Col(dbc.Input(
                        id='advanced_settings_duplicates',  type='number',  value=1.0, min=0, max=100, ), width=2
                        ),
                        dbc.Col(html.H6("(%) Allowed duplicates threshold [0,100]%"), align='center')]),

                        dbc.Row([#outlier threshold
                        dbc.Col(dbc.Input(
                        id='advanced_settings_outliers',  type='number',  value=1.0, min=0, max=100, ), width=2
                        ),
                        dbc.Col(html.H6("(%) Allowed outliers threshold [0,100]%"), align='center')]),

                        dbc.Row([#correlation threshold
                        dbc.Col(dbc.Input(
                        id='advanced_settings_correlation',  type='number',  value=0.9, min=0, max=1, ), width=2
                        ),
                        dbc.Col(html.H6("(fraction) Allowed correlation threshold [0,1]"), align='center')]),

                        dbc.Row([#duplicate columns
                        dbc.Col(dbc.Switch(
                        id='advanced_settings_duplicate_columns',  value = False,), width=2
                        ),
                        dbc.Col(html.H6("Allow duplicate columns"), align='center')]),

                        dbc.Row([#single value
                        dbc.Col(dbc.Switch(
                            id='advanced_settings_single_value', value=False, ), width=2
                        ),
                        dbc.Col(html.H6("Allow columns with the same unique value in each row (redundancy)"), align='center')]),

                        dbc.Row([#mixed data types
                        dbc.Col(dbc.Switch(
                        id='advanced_settings_mixed_data_types',  value = False,), width=2
                        ),
                        dbc.Col(html.H6("Allow mixed data types in columns"), align='center')]),

                        dbc.Row([#special characters
                        dbc.Col(dbc.Switch(
                            id='advanced_settings_special_characters', value=False, ), width=2
                        ),
                        dbc.Col(html.H6("Allow data points consisting of only special characters (!@#$%)"), align='center')]),

                        dbc.Row([#string mismatch
                        dbc.Col(dbc.Switch(
                        id='advanced_settings_string_mismatch',  value = False,), width=2
                        ),
                        dbc.Col(html.H6("Allow string mismatches (e.g. 'Red', 'RED' & 'red)'"), align='center')]),

                        dbc.Row([#conflicting labels
                        dbc.Col(dbc.Switch(
                            id='advanced_settings_conflicting_labels', value=False, ), width=2
                        ),
                        dbc.Col(html.H6("Allow conflicting labels (same feature values, different label)"), align='center')]),

                        html.Br(), #blank line
                        dbc.Row([
                        dbc.Col(dbc.Switch(id='advanced_settings_10k', value = True, label='Sample 10.000 rows for smooth running time'))]),

                ]
            ),
        ],start_collapsed=True,
        style={"width": "50%"},  # adjust as needed
    )

############################################################################################################################################
############################################################################################################################################
###################################################AUTOMATED CLEANING SECTION###############################################################
############################################################################################################################################
def layout_HITL_remediations(filepath, df_string_mismatch, df_special_characters, dtypes, dq_issues_dict):
    data = fetch_data(filepath)
    data_with_index = data.reset_index().rename(columns={'index': 'Original Index'}) #add original index column to keep track of original positioning
    filepath2 = filepath[:-4] +'_cleaned.pkl' #create a new file to not hinder the 'additional checks' when edits are made to the dataframe
    data_with_index.to_pickle(filepath2) #Store a new file so no problems occur when user decides to run the 'additional checks' after having cleaned his data

    df_string_mismatch, df_special_characters = fix_dataframes(df_string_mismatch, df_special_characters)


    dataframe_dict = {'df_string_mismatch': df_string_mismatch.to_dict(),
                      'df_special_characters': df_special_characters.to_dict()}
    dropdown_options_impute = ['most frequent', 'median', 'mean', 'do not impute'] #do not impute to let user choose a different constant value for each col
    impute_type_for_column = {} #storage for recommended imputation types (which are adjustable by dropdown)
    for col in data.columns:
        if dtypes[col] == 'floating' or dtypes[col] == 'integer' or dtypes[col] == 'numeric': #recommend mean
            impute_type_for_column[col] = 'mean'
        elif dtypes[col] == 'categorical' or dtypes[col] == 'boolean': #recommend most frequent
            impute_type_for_column[col] = 'most frequent'
        else:
            impute_type_for_column[col] = 'do not impute' #motivate users to think of an appropriate method themselves
    impute_type_for_column = pd.DataFrame(impute_type_for_column, index=[0])

    return html.Div([
        dbc.Row(
            [
                dbc.Col(dbc.Col(dbc.Button("5", color="secondary", outline=True, className="font-weight-bold",
                                       style={'fontSize': '20px', 'padding': '10px', 'width': '50px'}))),
                dbc.Col([
                    html.H3('Clean your dataset', style={'textAlign': 'center', "margin-right": "300px"}),
                    html.P(
                        'Accept & apply the recommended fixes with just a few clicks! Open one of the items below and the datatable '
               'will update accordingly. The datatable can be edited, and after you are finished cleaning your data, press "EXPORT CLEANED DATA" '
                        'to download the cleaned data.'
               ' No data in the datatable? Then there is nothing to fix for that issue!',
                        style={'textAlign': 'center', "margin-right": "200px"}),
                ], width=10, align='center'),
            ],
            justify="around"
        ),
        html.Div(id='hidden-div-duplicates', style={'display': 'none'}),
     html.Div(id='hidden-div-outliers', style={'display': 'none'}),
    dcc.Store(id='stored-filepath2', data=filepath2, storage_type='memory'),
     dcc.Store(id='store-deleted-indices', data=[]),
    dcc.Store(id='store-deleted-columns', data=[]),
     dcc.Store(id='store-changed-data', data=[]),
    dcc.Store(id='store-button-clicks', data={'missing_values' : 0, 'columns' : 0, 'string_mismatch' : 0, 'special_characters' : 0}),

    dcc.Store(id='store-dataframes', data=dataframe_dict),
        dbc.Accordion(
            [
                dbc.AccordionItem([
                        # Feature type table
                        dash_table.DataTable(
                            id='imputation-methods',
                            data=impute_type_for_column.to_dict('records'),
                            columns=[{"name": i, "id": i, 'presentation': 'dropdown'} for i in
                                     data.columns],
                            editable=True,
                            dropdown={
                                i: {'options': [{'label': j, 'value': j} for j in
                                                dropdown_options_impute]} for i in data.columns},
                            style_table={
                                'overflowX': 'scroll', 'height': '250px'
                            }
                        ),
                        dbc.Button("Impute missing values", id="impute-missing-values-button", color="danger", className="mr-1", n_clicks=0),
                        html.P('INFO: Imputations will only become visible in the datatable when switching tabs.')
                                ],
                    title="Missing value imputation ({})".format(len(dq_issues_dict['missing_values'])),
                    item_id="imputation",
                ),
                dbc.AccordionItem(
                    [
                    dbc.RadioItems(
                        options=[
                            {"label": "Duplicate instances", "value": 'duplicates'},
                            {"label": "Outlier instances", "value": 'outliers'},
                            {"label": "Redundant columns", "value": 'columns'}
                        ],
                        value='duplicates',
                        inline=True,
                        id="radioitems-deletions"
                    ),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("Delete duplicates", id="delete-duplicates-button", color="danger", className="mr-1", n_clicks=0),
                                width={"size": 2},  # for example, this could take up 2 of the 12 column units
                                align="start",  # align to the left
                                style={"display": "block"}
                            ),
                            dbc.Col(
                                dbc.Button("Delete outliers", id="delete-outliers-button", color="danger", className="mr-1", n_clicks=0),
                                width={"size": 2},  # for example, this could take up 2 of the 12 column units
                                align="start",  # align to the left
                                style={"display": "block"}
                            ),
                            dbc.Col([
                                dcc.Dropdown(
                                    id='column-dropdown',
                                    options=[{'label': i, 'value': i} for i in dq_issues_dict['redundant_columns']],
                                    value='Select a column to remove',
                                    style={"display": "none"}
                                )]),
                            dbc.Col(
                                dbc.Button("Delete column", id="delete-column-button", color="danger", n_clicks=0, className="mr-1", style={"display": "none"}),
                                width={"size": 2}
                            ),
                            dbc.Col(html.P('INFO: column deletions become visible when you switch to the next tab or download the dataset',
                               id='delete-column-text', style={"display": "block", "width": "16.66%"}),
                                    width={"size": 2}),
                            dbc.Col(
                                    width={"size": 6},
                                ),
                        ])

                    ],
                    title="Recommended deletions ({})".format(len(dq_issues_dict['duplicate_instances']) + len(dq_issues_dict['outlier_instances'])),
                    item_id="deletions",
                ),
                dbc.AccordionItem(
                    [
                        dbc.RadioItems(
                            options=[
                                {"label": "String mismatch", "value": 'string-mismatch'},
                                {"label": "Special characters", "value": 'special-characters'},
                                {"label": "Conflicting labels", "value": 'conflicting-labels'},
                            ],
                            value='string-mismatch',
                            inline=True,
                            id="radioitems-corrections"
                        ),
                            html.Hr(),
                        # dbc.Input(id="cap-column-value-outliers-input", inputmode='numeric', value="5000",
                        #           style={"display": "block", "width": "16.66%"}),
                        # dbc.Button("cap column value outliers", id="cap-column-value-outliers-button", color="danger",
                        #            className="mr-1", n_clicks=0,
                        #            style={"display": "block", "width": "16.66%"}),

                        # string mismatches button
                        dbc.Row([dbc.Col(
                                dcc.Dropdown(
                                    id='string-mismatch-dropdown',
                                    options=[{'label': i, 'value': i} for i in list(set(df_string_mismatch['Base form']))],
                                    value=df_string_mismatch['Base form'][0],

                                ), style={"display": "block", "width": "5%"}, id='string-mismatch-dropdown-col'),
                            dbc.Col(dbc.Button("Fix string mismatch", id="string-mismatch-button", color="danger",
                                   className="mr-1", n_clicks=0,
                                   ), style={"display": "None", "width": "16.66%"}, id="string-mismatch-button-col"),
                                ]),

                        # replace special character
                        dbc.Input(id="fix-special-characters-input", value="FILL IN REPLACEMENT VALUE",
                                  style={"display": "block", "width": "16.66%"}),
                        dbc.Button("Fix special characters", id="fix-special-characters-button", color="danger",
                                   className="mr-1", n_clicks=0,
                                   style={"display": "block", "width": "16.66%"}),

                        html.P('Walk through the table and adjust the target column values if required',
                               id='conflicting-labels-text', style={"display": "block", "width": "16.66%"}),


                            ],
                    title="Recommended corrections ({})".format(len(dq_issues_dict['string_mismatch'])
                                                                + len(dq_issues_dict['special_characters']) + len(dq_issues_dict['conflicting_labels'])),
                    item_id="corrections",
                ),
            ],
            id="accordion",
            start_collapsed=True
        ),
        html.Div(id='accordion-container'),
        dash_table.DataTable(
            id='datatable',
            columns=[{"name": i, "id": i} for i in data.columns],
            data=data_with_index.to_dict('records'),
            editable=True,
            row_deletable=True,
            page_action='native',
            page_current=0,
            page_size = 10,
            style_table={
                'overflowX': 'scroll'}
        ),
        dbc.Button('EXPORT CLEANED DATA', color = 'primary', id='export-button', n_clicks=0, className="mr-1",),
        dcc.Download(id="download")
    ])


@app.callback(
    Output('datatable', "data"),
    Output('datatable', 'page_action'),
    [Input("accordion", "active_item"),
     Input('radioitems-deletions', 'value'),
     Input('radioitems-corrections', 'value'),
     Input('delete-duplicates-button', 'n_clicks'),
    Input('delete-outliers-button', 'n_clicks'),
     Input('datatable', 'page_current'),
     Input('datatable', 'page_size'),
     State('stored-filepath2', 'data'),
     State('store-deleted-indices', 'data'),
    State('store-deleted-columns', 'data'),
     State('store-changed-data', 'data'),
     State('store-dq_issues_dict', 'data'),
     ]
)
def toggle_accordion(active_key, radio_deletions_selected, radio_corrections_selected, del_duplicates_button, del_outliers_button,
                     page_current, page_size,
                     filepath2, deleted_indices, deleted_columns, changed_data, dq_issues_dict):
    df = fetch_data(filepath2)
    df = update_dataframe(df, changed_data, deleted_indices, deleted_columns, filepath2) #apply changes to df when switch in accordion occurs
    if active_key == 'imputation':
        current_indices = obtain_current_indices(df, dq_issues_dict['missing_values'])
        df_filtered = df.iloc[current_indices]

        return df_filtered.to_dict('records'), 'custom'

    elif active_key == 'deletions':

        if radio_deletions_selected == 'duplicates':
            current_indices = obtain_current_indices(df, dq_issues_dict['duplicate_instances']) #TODO vervangen voor indices duplicate instances

            if del_duplicates_button: #if pressed, display empty df
                df_filtered = pd.DataFrame()

            else:
                df_filtered = df.iloc[current_indices]#filter rows based on the indices

            return df_filtered.to_dict('records'), 'custom'


        elif radio_deletions_selected == 'outliers':
            current_indices = obtain_current_indices(df, dq_issues_dict['outlier_instances'])  # TODO vervangen voor indices duplicate instances
            # filter rows based on the indices

            if del_outliers_button: #if pressed, display empty df
                df_filtered = pd.DataFrame()
            else:
                df_filtered = df.iloc[current_indices]#filter rows based on the indices

            #reset the data to show all rows
            return df_filtered.to_dict('records'), 'custom'


        elif radio_deletions_selected == 'columns':  #redundant columns selected
            #filter rows based on the indices
            df_filtered = df[:0]
            if deleted_columns:
                for column in list(set(deleted_columns)):
                    if column in df.columns:
                        df_filtered = df.drop(columns=column)

            return df_filtered.to_dict('records'), 'custom'

        else:
            return df.to_dict('records'), 'custom'

    elif active_key == 'corrections':

        if radio_corrections_selected == 'column-value-outliers':

            current_indices = obtain_current_indices(df, dq_issues_dict[
                'cv_outliers'])
            # filter rows based on the indices

            df_filtered = df.loc[current_indices]  # filter rows based on the indices

            # reset the data to show all rows
            return df_filtered.to_dict('records'), 'custom'

        elif radio_corrections_selected == 'string-mismatch':

            current_indices = obtain_current_indices(df, dq_issues_dict[
                'string_mismatch'])
            # filter rows based on the indices

            df_filtered = df.loc[current_indices]  # filter rows based on the indices

            # reset the data to show all rows
            return df_filtered.to_dict('records'), 'custom'


        elif radio_corrections_selected == 'special-characters':

            current_indices = obtain_current_indices(df, dq_issues_dict[
                'special_characters'])
            # filter rows based on the indices

            df_filtered = df.loc[current_indices]  # filter rows based on the indices

            # reset the data to show all rows
            return df_filtered.to_dict('records'), 'custom'


        elif radio_corrections_selected == 'conflicting-labels':

            current_indices = obtain_current_indices(df, dq_issues_dict[
                'conflicting_labels'])
            # filter rows based on the indices

            df_filtered = df.loc[current_indices]  # filter rows based on the indices

            # reset the data to show all rows
            return df_filtered.to_dict('records'), 'custom'

    else:
        return df.to_dict('records'), 'native' #use table style 'native' to speed up loading of page


@app.callback(
    [Output('store-deleted-indices', 'data'),
    Output('store-deleted-columns', 'data'),
     Output('store-changed-data', 'data'),
     Output('store-button-clicks', 'data')],
    Input('datatable', 'data_previous'),
    Input('datatable', 'data'),
    Input('impute-missing-values-button', 'n_clicks'),
    Input('delete-duplicates-button', 'n_clicks'),
    Input('delete-outliers-button', 'n_clicks'),
    Input('delete-column-button', 'n_clicks'),
    Input('string-mismatch-button', 'n_clicks'),
    Input('fix-special-characters-button', 'n_clicks'),
    [State('store-deleted-indices', 'data'),
    State('store-deleted-columns', 'data'),
     State('store-changed-data', 'data'),
    State('radioitems-deletions', 'value'),
    State('store-dq_issues_dict', 'data'),
    State('column-dropdown', 'value'),
    State('store-button-clicks', 'data'),
    State('string-mismatch-dropdown', 'value'),
    State('fix-special-characters-input', 'value'),
    State('dtypes_dropdown', 'data'),
    State('imputation-methods', 'data'),
    State('store-dataframes', 'data'),
    State("accordion", "active_item"),
    State('stored-filepath2', 'data'),
     ] #TODO: toevoegen dict met issues as state om huidige bug te voorkomen
)
def store_datatable_edits(previous, current, impute_missing_values_button, del_duplicates_button, del_outliers_button, del_column_button,
                          fix_string_mismatch_button, fix_special_characters_button,

                          deleted_indices, deleted_columns,
                          changed_data, radioitem_deletions, dq_issues_dict, column_dropdown_value, previous_button_clicks,
                          string_mismatch_dropdown_value, fix_special_characters_input, dtypes, imputation_methods, stored_dfs, active_item, filepath2):

    df_current, df_previous = pd.DataFrame(data=current), pd.DataFrame(data=previous)

    if df_current.empty:
        if active_item == 'deletions':
            if radioitem_deletions == 'duplicates':  # then the last item was removed in the datatable was removed
                deleted_indices.append(dq_issues_dict['duplicate_instances'])
                return deleted_indices, deleted_columns, changed_data, previous_button_clicks
            elif radioitem_deletions == 'outliers':  # then the last item was removed in the datatable was removed
                deleted_indices.append(dq_issues_dict['outlier_instances'])
                return deleted_indices, deleted_columns, changed_data, previous_button_clicks
            elif radioitem_deletions == 'columns':
                if del_column_button - previous_button_clicks[
                    'columns'] == 1:  # then it was clicked, and we should add the current column of the dropdown
                    deleted_columns.append(column_dropdown_value)
                    previous_button_clicks['columns'] += 1
                    return deleted_indices, deleted_columns, changed_data, previous_button_clicks
                else:
                    return deleted_indices, deleted_columns, changed_data, previous_button_clicks  # solely to prevent error in comparison of previous to current df


    if impute_missing_values_button - previous_button_clicks['missing_values'] == 1: #then it has been clicked
        previous_button_clicks['missing_values'] += 1
        df_entire = fetch_data(filepath2)
        dtypes_dict = dtypes[0]
        changed = dq_improvements.impute_missing_values(df_entire, dtypes_dict, imputation_methods) #use entire df for correct imputations
        changed_data.extend(changed)

    if fix_string_mismatch_button - previous_button_clicks['string_mismatch'] == 1: #then it has been clicked
        previous_button_clicks['string_mismatch'] += 1
        changed = dq_improvements.fix_string_mismatch(df_current, stored_dfs['df_string_mismatch'], string_mismatch_dropdown_value)
        changed_data.extend(changed)

    if fix_special_characters_button - previous_button_clicks['special_characters'] == 1: #then it has been clicked
        previous_button_clicks['special_characters'] += 1
        changed = dq_improvements.fix_special_characters(df_current, stored_dfs['df_special_characters'], fix_special_characters_input)
        changed_data.extend(changed)


    if previous is not None:
        if active_item == 'corrections' and df_current.empty:  # then the last item of the datable was removed using clicking
            deleted_index = df_previous.loc[0, 'Original Index']
            deleted_indices.append(deleted_index)
            return deleted_indices, deleted_columns, changed_data, previous_button_clicks

        common_indices = set(df_previous['Original Index']).intersection(set(df_current['Original Index']))
        #compare previous to current data to find the changed cells and deleted rows
        deleted_index = list(set(df_previous['Original Index']) - set(df_current['Original Index']))

        #check whether something was deleted, and only one row (if more rows were 'deleted', this indicates a switch in accordionitem)
        if deleted_index:
            if len(deleted_index) == 1: #TODO: toevoegen and in indices van dictionary met issues
                if deleted_index[0] in df_previous['Original Index'].values:
                    deleted_indices.append(deleted_index)

        df_current = df_current[df_current['Original Index'].isin(common_indices)]
        if 'level_0' in df_current.columns:
            df_current = df_current.drop(columns=['level_0'])
        df_current.reset_index(inplace=True)
        df_previous = df_previous[df_previous['Original Index'].isin(common_indices)]
        if 'level_0' in df_previous.columns:
            df_previous = df_previous.drop(columns=['level_0'])
        df_previous.reset_index(inplace=True)


        #if there are no deletions, check for changes
        if not deleted_index:
            diff_mask = (df_current != df_previous)
            diff_df = df_current[diff_mask]
            diff_series = diff_df.stack(dropna=True)
            if len(diff_series) == 1:
                # Iterate over the multi-index Series
                for idx, new_val in diff_series.iteritems():
                    row_idx, col_name = idx
                    row_idx_new = df_current.loc[row_idx, 'Original Index']
                    previous = df_previous.loc[row_idx, col_name]
                    change = {
                        "coordinates": (row_idx_new, col_name),
                        "new": new_val}
                        #"previous": previous}
                    changed_data.append(change)

        return deleted_indices, deleted_columns, changed_data, previous_button_clicks
    else:
        return deleted_indices, deleted_columns, changed_data, previous_button_clicks

def update_dataframe(df, changes, deleted_indices, deleted_columns, filepath2):
    #apply changes
    for change in changes:
        if not isinstance(change, str):
            coordinates = change['coordinates']
            new_val = change['new']
            orig_index = df.index[df['Original Index'] == coordinates[0]]
            if not orig_index.empty:
                current_index = orig_index.item() #obtain the current index
                df.at[current_index, coordinates[1]] = new_val

    #flatten list of deleted indices
    flattened_deleted_indices = [idx for sublist in deleted_indices for idx in (sublist if isinstance(sublist, list) else [sublist])]
    flattened_deleted_indices = list(set(flattened_deleted_indices))
    columns_to_drop = list(set(deleted_columns))

    #check which values that should be deleted still exist in the Original Index column
    mask = df['Original Index'].isin(flattened_deleted_indices)
    selected_rows = df[mask]
    indices_to_drop = list(selected_rows.index) #find the current index
    if indices_to_drop:
        df = df.drop(indices_to_drop)

    for column in columns_to_drop:
        if column in df.columns:
            df = df.drop(columns=column)

    if 'level_0' in df.columns:
        df = df.drop(columns=['level_0'])

    df = df.reset_index()

    df.to_pickle(filepath2)
    return df

@app.callback(
    Output('download', 'data'),
    Input('export-button', 'n_clicks'),
    [State('store-deleted-indices', 'data'),
    State('store-deleted-columns', 'data'),
     State('store-changed-data', 'data'),
     State('stored-filepath2', 'data')]
)
def export_dataset(n, deleted_indices, deleted_columns,changed_data, filepath2):
    if n:
        df = fetch_data(filepath2)
        df = update_dataframe(df, changed_data, deleted_indices, deleted_columns, filepath2)
        for col in ['level_0', 'index', 'Original Index']:
            if col in df.columns:
                df = df.drop(columns=col)
        return dcc.send_data_frame(df.to_excel, "cleaned_data.xlsx")



@app.callback(
    Output('delete-column-button', 'style'),
    Output('column-dropdown', 'style'),
    Output('delete-column-text', 'style'),
    [Input('radioitems-deletions', 'value')]
)
def show_column_deletions(radio_deletions_selected):
    if radio_deletions_selected == 'outliers' or radio_deletions_selected == 'duplicates':
        return {"display": "none"}, {"display": "none"}, {"display": "none"}  # show button
    else:
        return {"display": "block"}, {"display": "block"}, {"display": "block"}  # hide button

@app.callback(
    Output('delete-duplicates-button', 'style'),
    [Input('radioitems-deletions', 'value')]
)
def show_delete_duplicates_button(radio_corrections_selected):
    if radio_corrections_selected == 'duplicates':
        return {"display": "block"} # show button
    else:
        return {"display": "none"} # hide button

@app.callback(
    Output('delete-outliers-button', 'style'),
    [Input('radioitems-deletions', 'value')]
)
def show_delete_outliers_button(radio_corrections_selected):
    if radio_corrections_selected == 'outliers':
        return {"display": "block"} # show button
    else:
        return {"display": "none"} # hide button
# @app.callback(
#     Output('cap-column-value-outliers-input', 'style'),
#     Output('cap-column-value-outliers-button', 'style'),
#     [Input('radioitems-corrections', 'value')]
# )
# def show_delete_column_value_outliers_button(radio_corrections_selected):
#     if radio_corrections_selected == 'column-value-outliers':
#         return {"display": "block"}, {"display": "block"} # show button
#     else:
#         return {"display": "none"}, {"display": "none"} # hide button


@app.callback(
    Output('string-mismatch-button-col', 'style'),
    Output('string-mismatch-dropdown-col', 'style'),
    [Input('radioitems-corrections', 'value')]
)
def show_string_mismatch_button(radio_corrections_selected):
    if radio_corrections_selected == 'string-mismatch':
        return {"display": "block"}, {"display": "block"} # show button
    else:
        return {"display": "none"}, {"display": "none"} # hide button


@app.callback(
    Output('fix-special-characters-input', 'style'),
    Output('fix-special-characters-button', 'style'),
    [Input('radioitems-corrections', 'value')]
)
def show_special_characters(radio_corrections_selected):
    if radio_corrections_selected == 'special-characters':
        return {"display": "block"}, {"display": "block"} # show button
    else:
        return {"display": "none"}, {"display": "none"} # hide button


@app.callback(
    Output('conflicting-labels-text', 'style'),
    [Input('radioitems-corrections', 'value')]
)
def show_conflicting_labels_button(radio_corrections_selected):
    if radio_corrections_selected == 'conflicting-labels':
        return {"display": "block"} # show button
    else:
        return {"display": "none"} # hide button
def fetch_data(filepath):
    if os.path.exists(filepath):
        # If the file already exists, load the DataFrame from the cache
        df = pd.read_pickle(filepath)
        return df

def obtain_current_indices(df, original_indices):
    mask = df['Original Index'].isin(original_indices)
    selected_rows = df[mask]
    current_indices = selected_rows.index.tolist()
    return current_indices


def fix_dataframes(df_string_mismatch, df_special_characters):
    if 'Check notification' in list(df_string_mismatch.columns):
        df_string_mismatch = pd.DataFrame({
            'Base form': ['No string mismatches encountered'],
            'Value': ['No string mismatches encountered'],
        })

    if 'Check notification' in list(df_special_characters.columns):
        df_special_characters = pd.DataFrame({
        'Most Common Special-Only Samples': ["'No special characters found'"]
    })
    return df_string_mismatch, df_special_characters

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=False)
