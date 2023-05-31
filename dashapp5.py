import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, dcc, dash_table
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import datetime
import base64
from dash.dependencies import Input, Output, State
import arff
from sklearn.preprocessing import LabelEncoder
import missingno as msno
import plotly.figure_factory as ff
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
# Generate random dataset with missing values
np.random.seed(1)
data = np.random.randint(low=1, high=100, size=(10000, 5)).astype(float)
missing_indices = np.random.choice(data.size, size=int(data.size * 0.1), replace=False)
data.ravel()[missing_indices] = np.nan

# Create annotated heatmap
fig = ff.create_annotated_heatmap(data, colorscale=[[0, 'white'], [1, 'black']])

# Update layout
fig.update_layout(
    title='Matrix Plot with Missing Data',
    xaxis_title='Columns',
    yaxis_title='Rows'
)
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'Column 1': [1, 2, np.nan, 4, 5],
    'Column 2': [6, np.nan, 8, np.nan, 10],
    'Column 3': [11, 12, 13, np.nan, 15]
})

# Prepare data for plotting
x_data = df.columns
y_data = np.arange(len(df))

# Create the bar chart figure
fig = go.Figure()

for i, col in enumerate(x_data):
    values = df[col].values
    colors = ['white' if pd.isnull(val) else 'blue' for val in values]

    fig.add_trace(go.Bar(
        x=values,
        y=[i] * len(df),
        marker=dict(color=colors),
        orientation='h'
    ))

# Update the layout
fig.update_layout(
    title='Bar Chart with Missing Values',
    xaxis=dict(title='Values'),
    yaxis=dict(title='Indices', tickmode='array', tickvals=y_data, ticktext=y_data),
    showlegend=False
)

# Show the figure
fig.show()

# Display the figure

# Label encode the 'species' column using LabelEncoder from scikit-learn

# Create the PCP plot using Plotly Express

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

    html.H1(children='Hello Dash'),

    dcc.Graph(id='my_div', figure=fig)





])

if __name__ == '__main__':
    app.run_server(debug=True)
