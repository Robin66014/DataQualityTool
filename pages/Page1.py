import arff
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
#from app import dataset
from deepchecks.tabular.checks import ClassImbalance, ColumnsInfo
import io
import streamlit.components.v1 as components
from stqdm import stqdm
from time import sleep
from scipy.io.arff import loadarff
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, dcc, dash_table, callback
import io
import datetime
import base64
from dash.dependencies import Input, Output, State
from DataTypeInference import obtain_feature_type_table, createDatasetObject
import dash
from dash import dcc, html
import plotly.express as px
import settings


dash.register_page(__name__, path='/')
#df = pd.read_csv('datasets\Iris.csv')
sortingHatInf_datatypes = ['not-generalizable', 'floating', 'integer', 'categorical', 'boolean', 'datetime', 'sentence', 'url',
                           'embedded-number', 'list', 'context-specific', 'numeric']

button_style = {'background-color': 'blue',
                    'color': 'white',
                    'height': '50px',
                    'margin-top': '50px',
                    'margin-left': '50px'}
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

layout = html.Div([
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
    dcc.Location(id='url', refresh=False),
    html.Div(id='output-div'),
    html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename, date):
    """"Dash function belonging to the file upload button for reading csv and xlsx files (https://dash.plotly.com/dash-core-components/upload)"""
    #settings.init()
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

        #TODO: werkend krijgen voor alle bestandtypes
        featureTypeTable = obtain_feature_type_table(df)
        settings.uploaded_file_df = df
        settings.first_time = True
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.P('Uploaded file: {}'.format(filename)),
        #html.H6(datetime.datetime.fromtimestamp(date)),
        html.H6('Choose your target column'),
        dcc.Dropdown(id="targetColumn", options=[{'label':'None', 'value':'None'}] + [{'label':x, 'value':x} for x in df.columns], value = 'None'),
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
                'overflowX': 'scroll'
            }
        ),
        dcc.Link(html.Button('Run data quality checks', style=button_style), href=dash.page_registry['pages.Page2']['path']),
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
        )
    ])
@callback(
    [Output('dd-output-container', 'children'), Output('output-div', 'children')],
    [Input('targetColumn', 'value')]#, Input('submit-button','n_clicks'), State('targetColumn','targetvalue')]
)
def update_output(value):#, n, target):
    return f'You have selected {value}, as your target column', dash.no_update


@callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
