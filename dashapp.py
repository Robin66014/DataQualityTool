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
import label_purity

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
    html.Div(id='container-checks-button-pressed'),




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
        settings.first_time = True
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
        dash_table.DataTable(
            featureTypeTable.to_dict('records'),
            columns = [{"name": i, "id": i, 'presentation': 'dropdown'} for i in featureTypeTable.columns],
            editable=True,
            dropdown={
                i: {'options': [{'label': j, 'value': j} for j in
                                       sortingHatInf_datatypes]} for i in
                featureTypeTable.columns},
            style_table={
                'overflowX': 'scroll'}
        ),
        html.Button('Run data quality checks', id='run-checks-button', n_clicks=0, style=button_style),
        html.Hr(),  # horizontal line
        html.H6('Dataset overview'),
        html.P('Click the dataset to edit a cell, press the export button to download the edited dataset'),

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
    [Input('run-checks-button', 'n_clicks'), Input('stored-data', 'data')]
)
def run_checks(n_clicks, df_json):#, n, target):
    if n_clicks >= 1: #run checks button clicked
        df = pd.DataFrame(df_json) #deserialize JSON string stored in web browser

        #Running of the checks
        df_missing_values = duplicates_and_missing.missing_values(df)
        df_amount_of_diff_values = type_integrity.amount_of_diff_values(df)
        df_mixed_data_types = type_integrity.mixed_data_types(df)
        #df_special_characters = type_integrity.special_characters(df)
        #df_string_mismatch = type_integrity.string_mismatch(df)

        #TODO: duplicates check
        return html.Div([dmc.Accordion(
            children=[
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Duplicates & missing values ({})".format(36)),
                        dmc.AccordionPanel([dash_table.DataTable(df_missing_values.to_dict('records')),
                            "Colors, fonts, shadows and many other parts are customizable to fit your design needs"]
                        ),

                    ],
                    value="duplicatesandmissing",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Type integrity ({})".format(36)),
                        dmc.AccordionPanel([
                            # dash_table.DataTable(df_amount_of_diff_values.to_dict('records')),
                            #                 dash_table.DataTable(df_mixed_data_types.to_dict('records')),
                            #                 dash_table.DataTable(df_special_characters.to_dict('records')),
                            #                 dash_table.DataTable(df_string_mismatch.to_dict('records')),
                            "Colors, fonts, shadows and many other parts are customizable to fit your design needs"]
                        ),
                    ],
                    value="typeintegrity",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Outliers & correlations ({})".format(36)),
                        dmc.AccordionPanel(
                            "Configure temp appearance and behavior with vast amount of settings or overwrite any part of "
                            "component styles "
                        ),
                    ],
                    value="outliersandcorrelations",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Label purity ({})".format(36)),
                        dmc.AccordionPanel(
                            "Configure temp appearance and behavior with vast amount of settings or overwrite any part of "
                            "component styles "
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





# TODO: oude code, checkbox
# app.layout = html.Div([
#
#     html.H1("Data quality web app", style={'text-align': 'center'}),
#
#     dcc.Dropdown(id="slct_year",
#                  options=[
#                      {"label": "2015", "value": 2015},
#                      {"label": "2016", "value": 2016},
#                      {"label": "2017", "value": 2017},
#                      {"label": "2018", "value": 2018}],
#                  multi=False,
#                  value=2015,
#                  style={'width': "40%"}
#                  ),
#
#     html.Div(id='output_container', children=[]),
#     html.Br(),
#
#     #dcc.Graph(id='my_bee_map', figure={})
#
# ])
#
#
#
# if __name__ == '__main__':
#     app.run_server(debug=True)