import base64
import datetime
import io
import scipy.io.arff as arff
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
import matplotlib as plt
import missingno as msno
import os
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Button('Generate Graph', id='generate-button', n_clicks=0,),
    html.Img(id='example', alt="my image", width="750", height="500"),
    #html.Div(id='example'),
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
        elif '.arff' in filename:
            # Assume that the user uploaded an ARFF file
            data = io.StringIO(decoded.decode('utf-8'))
            print('@@@@data',data)
            arff_data = arff.loadarff(data)
            print('@@@@arff_data', arff_data[0])
            #print(arff_data[0])
            df = pd.DataFrame(arff_data[0])
            for column in df.columns:
                if df[column].dtype == object:  # Check if the column contains object data type (usually used for strings)
                    df[column] = df[column].str.decode('utf-8', errors='ignore')
        elif '.parquet' in filename:
            # Assume that the user uploaded a Parquet file
            df = pd.read_parquet(io.BytesIO(decoded))
        elif '.json' in filename: #TODO: hier verdergaan
            # Assume that the user uploaded a JSON file
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
        elif '.feather' in filename:
            # Assume that the user uploaded a Feather file
            df = pd.read_feather(io.BytesIO(decoded))


    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        # For debugging, display the raw contents provided by the web browser
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line


    ])

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
    dash.dependencies.Output('example', 'src'),
    [Input('generate-button', 'n_clicks')]
)
def update_figure(n_points):
    #create some matplotlib graph
    if n_points >= 1:

        df = pd.read_csv('datasets/creditcard.csv')
        fig = msno.matrix(df)
        fig_copy = fig.get_figure()
        fig_copy.savefig('assets/missingno_plot.png', bbox_inches='tight')
        #img_source = os.path.join(current_path, 'cached_files/missingno_plot.png')
        img_source = 'assets/missingno_plot.png'
        print(img_source)

        return img_source

if __name__ == '__main__':
    app.run_server(debug=True)