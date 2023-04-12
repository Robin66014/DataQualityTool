import pandas as pd
import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import MixedDataTypes
import pandas as pd
from sklearn.datasets import load_iris
from deepchecks.tabular.datasets.classification.phishing import load_data
import numpy as np
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import cleanlab
from cleanlab.classification import CleanLearning
from sklearn.linear_model import LogisticRegression
from scipy.io.arff import loadarff

# data = loadarff('datasets/Iris.csv')
# #df = pd.read_csv('datasets/Iris.csv')
# df = pd.DataFrame(data[0])
# io.StringIO(decoded.decode('utf-8'))
# print(df.head())
# df = pd.read_csv('datasets\data.csv')
# profile = ProfileReport(df, title="Profiling Report")
#
# print(profile.html)
# #profile.to_file("your_report.html")
# #As a JSON string
# json_data = profile.to_json()
# print(json_data)

# app.layout = html.Div([
#     dcc.Dropdown(
#         id = 'dropdown-to-show_or_hide-element',
#         options=[
#             {'label': 'Show element', 'value': 'on'},
#             {'label': 'Hide element', 'value': 'off'}
#         ],
#         value = 'on'
#     ),
#
#     # Create Div to place a conditionally visible element inside
#     html.Div([
#         # Create element to hide/show, in this case an 'Input Component'
#         dcc.Input(
#         id = 'element-to-hide',
#         placeholder = 'something',
#         value = 'Can you see me?',
#         )
#     ], style= {'display': 'block'} # <-- This is the line that will be changed by the dropdown callback
#     )
#     ])

from ydata_synthetic import streamlit_app

streamlit_app.run()