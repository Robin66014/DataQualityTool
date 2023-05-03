import pandas as pd
import sortinghatinf #algorithm to predict the feature type out og [numeric, categorical, datetime, sentence,
from deepchecks.tabular import Dataset
from deepchecks.tabular.datasets.classification import adult

#data = pd.read_csv('datasets\Iris.csv')
def obtain_feature_type_table(df):
    """"Function to predict the feature types
    INPUT: pandas DataFrame
    OUTPUT: Pandas DataFrame with feature types at index 0 and nothing else : [column names ]
                                                                              [feature_types]
    """
    #print('df.columns:', list(df.columns))
    predicted_feature_types = sortinghatinf.get_expanded_feature_types(df)
    feature_type_table = pd.DataFrame(columns=list(df.columns))
    feature_type_table.loc[len(feature_type_table)] = predicted_feature_types

    return feature_type_table


def createDatasetObject(df, feature_types, label):
    #obtain catgeorical feature names
    categorical_features = []
    for key, value in feature_types.items():
        if value == 'categorical' and key != label:
            categorical_features.append(key)
    #date_name = #todo hoe date_time en ID kolom meenemen in Dataset object? Gewoon weglaten in het begin?

    if label != 'None':
        if df[label].nunique() == 2: #binary classification
            ds = Dataset(df, label=label, cat_features=categorical_features, label_type='binary')
        elif df[label].nunique() > 2 and df[label].dtype == 'object': #likely a multi-class classifcation problem
            ds = Dataset(df, label=label, cat_features=categorical_features, label_type='multiclass')
        else: #likely a regression problem
            ds = Dataset(df, label=label, cat_features=categorical_features, label_type='regression')

    else: #no label selected by app user
        ds = Dataset(df, cat_features=categorical_features)

    return ds
