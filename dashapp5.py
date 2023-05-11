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
import testingFile
import dash_mantine_components as dmc

from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

#df = pd.read_csv('datasets\Iris.csv')
sortingHatInf_datatypes = ['not-generalizable', 'floating', 'integer', 'categorical', 'boolean', 'datetime', 'sentence', 'url',
                           'embedded-number', 'list', 'context-specific', 'numeric']
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

button_style = {'background-color': 'blue',
                    'color': 'white',
                    'height': '50px',
                    'margin-top': '50px',
                    'margin-left': '50px'}
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app = dash.Dash(__name__, suppress_callback_exceptions=False)
app.title = "Data quality toolkit"

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
df = pd.read_csv('datasets/iris.csv')

# Label encode the 'species' column using LabelEncoder from scikit-learn
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['Species'])

# Create the PCP plot using Plotly Express
fig = px.parallel_coordinates(df, color='species_encoded')

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

    dcc.Graph(figure=fig)





])



if __name__ == '__main__':
    app.run_server(debug=True)
