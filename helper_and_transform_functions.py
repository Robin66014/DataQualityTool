import pandas as pd
from scipy.stats import shapiro, anderson, kstest, normaltest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import shapiro, anderson, kstest, normaltest
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import preprocessing

def test_normality(dataset, column_types):
    """
    Performs normality tests on all numeric columns of a dataset and returns the test results.

    Args:
        dataset (pandas DataFrame): The input dataset to test.
        column_types (list): A list of column data types for the dataset.

    Returns:
        pandas DataFrame: A dataframe with the test results for all numeric columns.
    """
    # Create an empty dataframe to store the test results
    test_results = pd.DataFrame(columns=['column', 'shapiro_wilk_stat', 'shapiro_wilk_pvalue',
                                         'anderson_stat', 'anderson_crit_vals', 'anderson_sig_levels',
                                         'kolmogorov_smirnov_stat', 'kolmogorov_smirnov_pvalue',
                                         'd_agostino_pearson_stat', 'd_agostino_pearson_pvalue'])

    # Iterate through each column in the dataset
    for col in dataset.columns:
        if column_types[col] == 'numeric':
            data = dataset[col]

            #statistical tests
            shapiro_result = shapiro(data)
            anderson_result = anderson(data)
            ks_result = kstest(data, 'norm')
            d_ap_result = normaltest(data)
            #put in df
            test_results = test_results.append({'column': col,
                                                'shapiro_wilk_stat': shapiro_result.statistic,
                                                'shapiro_wilk_pvalue': shapiro_result.pvalue,
                                                'anderson_stat': anderson_result.statistic,
                                                'anderson_crit_vals': anderson_result.critical_values,
                                                'anderson_sig_levels': anderson_result.significance_level,
                                                'kolmogorov_smirnov_stat': ks_result.statistic,
                                                'kolmogorov_smirnov_pvalue': ks_result.pvalue,
                                                'd_agostino_pearson_stat': d_ap_result.statistic,
                                                'd_agostino_pearson_pvalue': d_ap_result.pvalue},
                                               ignore_index=True)
    return test_results


def check_dimensionality():

    return None


def encode_categorical_columns(dataset, target, data_types):
    # Find all categorical columns
    # TODO: regel hieronder aanpassen naar wat de user heeft ingegeven (let hierbij op strings als dtype, moeten gezien worden als categorical)
    categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()

    target_is_categorical = False
    if target != 'None':
        # remove target as we want to label encode this (for classification problems)
        if target in categorical_cols:
            target_is_categorical = True
            categorical_cols.remove(target)

            # label encode target
            le = LabelEncoder()
            encoded_target = le.fit_transform(dataset[target])
            # replace target column with label encoded values
            dataset.drop(columns=[target], inplace=True)
            dataset[target] = encoded_target

    if not categorical_cols:  # then no features are categorical, and we're done
        return dataset

    # if there are categorical columns, we want to one-hot-encode them

    # encode categoricals
    encoder = OneHotEncoder(handle_unknown='ignore', max_categories=100)
    encoded_columns = encoder.fit_transform(dataset[categorical_cols])
    new_columns = pd.DataFrame(encoded_columns.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

    # add new columns to df and drop old ones
    dataset_encoded = pd.concat([dataset, new_columns], axis=1)
    dataset_encoded = dataset_encoded.drop(columns=categorical_cols)

    # reposition target column to the end of the dataframe
    if target != 'None' and target_is_categorical:
        dataset_encoded.drop(columns=[target], inplace=True)
        dataset_encoded[target] = encoded_target

    #XGBClassifier doesn't accept: [, ] or <, so loop over the columns and change the names if they contain such values
    new_col_names = {col: col.replace('<', '(smaller than)').replace('[', '(').replace(']', ')') for col in dataset_encoded.columns}
    dataset_encoded = dataset_encoded.rename(columns=new_col_names)

    return dataset_encoded

def pcp_plot():

    return None

def missingno_plot():

    return None

def plot_feature_importance(X, y, feature_names):
    """
    Predicts the feature importance using a random forest and plots it.

    Parameters:
    -----------
    X : numpy array, shape (n_samples, n_features)
        The input data.
    y : numpy array, shape (n_samples,)
        The target values.
    feature_names : list of strings, shape (n_features,)
        The names of the features.

    Returns:
    --------
    None
    """
    # Initialize the random forest regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the random forest to the data
    rf.fit(X, y)

    # Get the feature importances
    feature_importances = rf.feature_importances_

    # Sort the feature importances in descending order
    indices = np.argsort(feature_importances)[::-1]

    # Plot the feature importances
    plt.figure()
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), feature_importances[indices])
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.show()




# Generate some random data
X = np.random.rand(100, 5)
y = np.random.rand(100)

# Define the feature names
feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]

# Call the function to plot the feature importance
plot_feature_importance(X, y, feature_names)

