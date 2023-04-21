# import pandas as pd
# import streamlit as st
# from deepchecks.tabular import Dataset
# from deepchecks.tabular.checks import MixedDataTypes
# import pandas as pd
# from sklearn.datasets import load_iris
# from deepchecks.tabular.datasets.classification.phishing import load_data
# import numpy as np
# import numpy as np
# import pandas as pd
# from ydata_profiling import ProfileReport
# import cleanlab
# from cleanlab.classification import CleanLearning
# from sklearn.linear_model import LogisticRegression
# from scipy.io.arff import loadarff

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

#from ydata_synthetic import streamlit_app
# import aif360
# import fairlearn
# from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
# from sklearn.metrics import accuracy_score
# import pandas as pd
# y_true = [1, 2, 1, 1, 4, 0, 0, 1, 1, 0]
# y_pred = [0, 1, 1, 1, 3, 0, 0, 0, 1, 1]
# sex = ['Female'] * 5 + ['Male'] * 5
# metrics = {"accuracy":accuracy_score, "selection_rate": selection_rate}
# mf1 = MetricFrame(metrics = metrics,y_true = y_true,y_pred = y_pred,sensitive_features = sex)
# print(mf1.by_group)
# #streamlit_app.run()
#
# dpd = demographic_parity_difference(y_true = y_true,y_pred = y_pred,sensitive_features = sex)
# print(dpd)
# print(mf1.difference())
# Import necessary libraries and functions
# from aif360.datasets import AdultDataset
# from aif360.algorithms.postprocessing import EqOddsPostprocessing
# from aif360.metrics import ClassificationMetric
# from sklearn.linear_model import LogisticRegression
#
# # Load the Adult dataset
# dataset = AdultDataset()
#
# # Split the dataset into training and testing sets
# train, test = dataset.split([0.7], shuffle=True)
#
# # Define the logistic regression classifier
# lr = LogisticRegression()
#
# # Train the classifier on the training set
# lr.fit(train.features, train.labels.ravel())
#
# # Make predictions on the testing set
# test_pred = lr.predict(test.features)
#
# # Define the classification metric for equalized odds
# metric = ClassificationMetric(test, test_pred, privileged_groups=[{'sex': 1}], unprivileged_groups=[{'sex': 0}])
#
# # Calculate the equalized odds difference for multiple subgroups
# eq_opp = EqOddsPostprocessing(unprivileged_groups=[{'race': 2}, {'race': 3}, {'race': 4}])
# eq_opp.fit(test, test_pred, metric)
# test_pred_eq = eq_opp.predict(test_pred, test)
#
# # Calculate the new classification metric for equalized odds
# metric_eq = ClassificationMetric(test, test_pred_eq, privileged_groups=[{'sex': 1}], unprivileged_groups=[{'sex': 0}])
# print('Equalized odds difference: %.4f' % metric_eq.equal_opportunity_difference())

# df1 = pd.DataFrame({'A':['yes','yes','yes','yes','no','no','yes','yes','yes','no'],
#                    'B': ['yes','no','no','no','yes','yes','no','yes','yes','no']})
# df1 = pd.read_csv('datasets\Iris.csv')
# cl = df1.columns
# print(df1.groupby(list(cl)).size().reset_index().rename(columns={0:'count'}))

# import pandas_dq
# data = pd.read_csv('datasets\Iris.csv')
# report = pandas_dq.dq_report(data, target=None, csv_engine="pandas", verbose=1).to_html()
# print(report)
import cleanlab
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def main():
    # your code here
    df = pd.read_csv('datasets\Iris.csv')
    # Create a label encoder object
    le = LabelEncoder()
    # Fit and transform the "Species" column
    df['Species'] = le.fit_transform(df['Species'])
    # cleanlab works with **any classifier**. Yup, you can use PyTorch/TensorFlow/OpenAI/XGBoost/etc.
    cl = cleanlab.classification.CleanLearning(RandomForestClassifier())
    data = df.loc[:, df.columns != 'Species']

    # cleanlab finds data and label issues in **any dataset**... in ONE line of code!
    # Fit model to messy, real-world data, automatically training on cleaned data.
    _ = cl.fit(data, list(df['Species']))
    # See the label quality for every example, which data has issues, and more.
    print(cl.get_label_issues().head())

if __name__ == '__main__':
    main()



