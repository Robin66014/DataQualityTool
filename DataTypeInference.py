import pandas as pd
import sortinghatinf #algorithm to predict the feature type out og [numeric, categorical, datetime, sentence,
# url, embedded-number, list, not-generalizable, context-specific]

data = pd.read_csv('datasets\Iris.csv')
def obtain_feature_type_table(df):
    """"Function to predict the feature types
    INPUT: pandas DataFrame
    OUTPUT: Pandas DataFrame with feature types at index 0 and nothing else : [column names ]
                                                                              [feature_types]
    """
    #print('df.columns:', list(df.columns))
    predicted_feature_types = sortinghatinf.get_expanded_feature_types(df)
    print(predicted_feature_types)
    new_df = pd.DataFrame(columns=list(df.columns))
    new_df.loc[len(new_df)] = predicted_feature_types

    return new_df
