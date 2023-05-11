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
from sklearn.preprocessing import LabelEncoder

#PROBLEM CAUSE STARTS HERE
# from duplicates_and_missing import missing_values, duplicates
from type_integrity import amount_of_diff_values, mixed_data_types, special_characters, string_mismatch
from outliers_and_correlations import feature_label_correlation, feature_feature_correlation, outlier_detection
#from label_purity import class_imbalance, conflicting_labels
#from plot_and_transform_functions import pandas_dq_report, encode_categorical_columns, pcp_plot, missingno_plot, plot_dataset_distributions
#END OF PROBLEM CAUSE


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

df = pd.read_csv('datasets/iris.csv')

# Label encode the 'species' column using LabelEncoder from scikit-learn
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['Species'])

# Create the PCP plot using Plotly Express
fig = px.parallel_coordinates(df, color='species_encoded')

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

if __name__ == '__main__':
    app.run_server(debug=True)
