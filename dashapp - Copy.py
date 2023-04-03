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



#df = pd.read_csv('datasets\Iris.csv')
sortingHatInf_datatypes = ['not-generalizable', 'floating', 'integer', 'categorical', 'boolean', 'datetime', 'sentence', 'url',
                           'embedded-number', 'list', 'context-specific', 'numeric']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True, use_pages=True)

app.layout = html.Div([
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
    html.Div(id='dd-output-container'),
    html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        featureTypeTable = obtain_feature_type_table(df)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),
        html.P('Choose your target column'),
        dcc.Dropdown(id="targetColumn", options=[{'label':'None', 'value':'None'}] + [{'label':x, 'value':x} for x in df.columns], value = 'None'),
        html.Button(id='submit-button', children='Run checks'),
        dash_table.DataTable(
            featureTypeTable.to_dict('records'),
            columns = [{"name": i, "id": i, 'presentation': 'dropdown'} for i in featureTypeTable.columns],
            editable=True,
            dropdown={
                i: {'options': [{'label': j, 'value': j} for j in
                                       sortingHatInf_datatypes]} for i in
                featureTypeTable.columns},
        ),
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            page_size = 10,
            editable=True,
            export_format='csv',
        ),
        #dcc.Store(id='stored-data', data = df.to_dict('records')),
        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
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

# @app.callback(Output('output-div', 'children'),
#               Input('submit-button','n_clicks'),
#               State('targetColumn','targetvalue'))
# def make_graphs(n, data):
#
#     if n is None:
#         data = data
#         return dash.no_update


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