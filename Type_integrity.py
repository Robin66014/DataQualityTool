import pandas
import streamlit as st
from deepchecks.tabular import Dataset
import deepchecks.tabular.checks
import pandas as pd
from sklearn.datasets import load_iris
from deepchecks.tabular.datasets.classification.phishing import load_data
import numpy as np
data = pd.read_csv('datasets\Iris.csv')
#dataset = Dataset(data, label = 'Species')
amount_of_columns = float('inf')
amount_of_samples = float('inf')









data = {'col1': [pd.NA, pd.NaT], 'col2': ['test', pd.NaT], 'col3': ['1', 'cat']}
dataframe = pd.DataFrame(data=data)
missing_values(dataframe)

res = pandas.NA == 'na'
print(res)
