import pandas as pd
import sortinghatinf #algorithm to predict the feature type out og [numeric, categorical, datetime, sentence,
from deepchecks.tabular import Dataset


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
    dict_feature_types = feature_types[0]
    #obtain catgeorical feature names
    categorical_features = []
    for key, value in dict_feature_types.items():
        if value == 'categorical' and key != label:
            categorical_features.append(key)
    #date_name = #todo hoe date_time en ID kolom meenemen in Dataset object? Gewoon weglaten in het begin?

    if label != 'None':
        ds = Dataset(df, label=label, cat_features=categorical_features)
    else: #no label selected by app user
        ds = Dataset(df, cat_features=categorical_features)

    return ds

# def createDatasetObject(df, featureTypeTable, label):
#     feature_types = featureTypeTable.loc[len(featureTypeTable)-1, :].values.tolist()
#     column_names = featureTypeTable.columns.values.tolist()
#
#     #obtain catgeorical feature names
#     categorical_features = []
#     index = 0
#     for feature in feature_types:
#         if feature == 'categorical' and column_names[index] != label:
#             categorical_features.append(column_names[index])
#         index += 1
#
#     #date_name = #todo hoe date_time en ID kolom meenemen in Dataset object? Gewoon weglaten in het begin?
#
#     if label != 'NOT APPLICABLE':
#         ds = Dataset(df, label=label, cat_features=categorical_features)
#     else: #no label exists in dataset
#         ds = Dataset(df, cat_features=categorical_features)
#
#     return ds


#df = obtain_feature_type_table(data)

# print(data.head())

