import pandas as pd
import sortinghatinf #algorithm to predict the feature type out og [numeric, categorical, datetime, sentence,
from deepchecks.tabular import Dataset
from plot_and_transform_functions import dash_datatable_format_fix

def obtain_feature_type_table(df):
    """"Function to predict the data types of the features in the df
    """
    #print('df.columns:', list(df.columns))
    predicted_feature_types = sortinghatinf.get_sortinghat_types(df)
    feature_type_table = pd.DataFrame(columns=list(df.columns))
    feature_type_table.loc[len(feature_type_table)] = predicted_feature_types
    feature_type_table = dash_datatable_format_fix(feature_type_table)
    return feature_type_table


def createDatasetObject(df, dtypes, target):
    """"Creates Deepchecks dataset object for more accurate analysis than with dataframes"""
    #obtain catgeorical feature names
    categorical_features = []
    for key, value in dtypes.items():
        if (value == 'categorical' or value == 'boolean') and key != target:
            categorical_features.append(key)
    #check type of ML task
    if target != 'None':
        if df[target].nunique() == 2: #binary classification
            ds = Dataset(df, label=target, cat_features=categorical_features, label_type='binary')
        elif df[target].nunique() > 2 and (dtypes[target] == 'categorical' or dtypes[target] == 'boolean'): #likely a multi-class classifcation problem
            ds = Dataset(df, label=target, cat_features=categorical_features, label_type='multiclass')
        else: #likely a regression problem
            ds = Dataset(df, label=target, cat_features=categorical_features, label_type='regression')

    else: #no label selected by app user
        ds = Dataset(df, cat_features=categorical_features)

    return ds
