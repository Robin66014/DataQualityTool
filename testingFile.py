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

data = pd.read_csv('datasets\Iris.csv')


df = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
profile = ProfileReport(df, title="Profiling Report")
json_data = profile.to_json()
print(json_data)
profile.to_file("your_report.json")