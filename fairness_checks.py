
#TODO: Later implementeren
import pandas as pd
from itertools import product
import plotly.express as px

def sensitive_feature_combinations(dataset_original, sensitive_features, target_column, bins=5):
    """"function that finds all combinations of possible sensitive features and displays them in a table,
     used for plotting the stacked bar chart later on"""

    dataset = dataset_original.copy()
    #TODO: callback maken die sensitive features ingeeft
    #TODO: iets bedenken voor regression
    #bin numeric sensitive features into 5 bins
    for feat in sensitive_features:
        if pd.api.types.is_numeric_dtype(dataset[feat]):        #TODO: baseren op data type inference sortinghat
            dataset[feat] = pd.cut(dataset[feat], bins=bins)

    #obtain all combinations of sensitive features
    sensitive_combinations = list(product(*[dataset[feat].unique() for feat in sensitive_features]))
    #print(sensitive_combinations)

    counts = {}
    # Loop through each sensitive feature combination
    for combo in sensitive_combinations:
        #create mask for rows that match the combination
        mask = True
        for i, feat in enumerate(sensitive_features):
            mask = mask & (dataset[feat] == combo[i])

        #count combinations per target
        count = dataset.loc[mask, target_column].value_counts()
        counts[combo] = count.to_dict()

    result = pd.DataFrame.from_dict(counts, orient='index').fillna(0)

    #add columns count & sensitive feautures
    result['count'] = result.sum(axis=1)
    result['sensitive_features'] = result.index.map(lambda x: ', '.join(map(str, x)))
    #reorder columns
    result = result[['sensitive_features', 'count'] + list(dataset[target_column].unique())]

    return result




def plot_stacked_barchart(sensitive_feature_counts_table):
    """"creates a plotly stacked bar chart to display the distribution of class labels per sensitive subgroup"""
    #table always looks the same (sensitive features (vary in amount), count, class labels (vary in amount))
    list_of_labels = list(sensitive_feature_counts_table.columns)
    list_of_labels.remove('sensitive_features')
    list_of_labels.remove('count')
    fig = px.bar(sensitive_feature_counts_table, x=list_of_labels, y='sensitive_features', barmode='stack', title='Sensitive Feature Combinations vs Target Column', orientation='h')
    # Set the x-axis title
    fig.update_xaxes(title_text='Count')

    return fig

