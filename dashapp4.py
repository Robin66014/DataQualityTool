import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('datasets/iris.csv')

# Label encode the 'species' column using LabelEncoder from scikit-learn
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['Species'])

# Create the PCP plot using Plotly Express
fig = px.parallel_coordinates(df, color='species_encoded')
# Set up the Dash app
app = dash.Dash(__name__)

# Create the app layout
app.layout = html.Div([dcc.Upload(
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)