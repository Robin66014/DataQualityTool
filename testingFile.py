import pandas as pd
import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import MixedDataTypes
import pandas as pd
from sklearn.datasets import load_iris
from deepchecks.tabular.datasets.classification.phishing import load_data
import numpy as np
data = pd.read_csv('datasets\Iris.csv')

adult_dataset = Dataset(adult_df, cat_features=['workclass', 'education'])
check = MixedDataTypes()
result = check.run(adult_dataset)
print('@@@@', type(result)) #returns check_result.CheckResult item
#result = IsSingleValue().run(dataset)
#result.save_as_html()
res = result.value #returns dict
print('@@@', type(res))
#checkJson = result.to_json()
#checkJson = check.reduce_output()
#print(checkJson)


print(lst)