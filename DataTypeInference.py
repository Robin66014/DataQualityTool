import pandas as pd
import sortinghatinf #algorithm to predict the feature type out og [numeric, categorical, datetime, sentence,
from deepchecks.tabular import Dataset
from plot_and_transform_functions import dash_datatable_format_fix
from deepchecks.tabular.datasets.classification import adult

#data = pd.read_csv('datasets\Iris.csv')
def obtain_feature_type_table(df):
    """"Function to predict the feature types
    INPUT: pandas DataFrame
    OUTPUT: Pandas DataFrame with feature types at index 0 and nothing else : [column names ]
                                                                              [feature_types]
    """
    #print('df.columns:', list(df.columns))
    predicted_feature_types = sortinghatinf.get_sortinghat_types(df)
    feature_type_table = pd.DataFrame(columns=list(df.columns))
    feature_type_table.loc[len(feature_type_table)] = predicted_feature_types
    feature_type_table = dash_datatable_format_fix(feature_type_table)
    return feature_type_table


def createDatasetObject(df, dtypes, target):
    #obtain catgeorical feature names
    categorical_features = []
    for key, value in dtypes.items():
        if (value == 'categorical' or value == 'boolean') and key != target:
            categorical_features.append(key)
    #date_name = #todo hoe date_time en ID kolom meenemen in Dataset object? Gewoon weglaten in het begin?
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
