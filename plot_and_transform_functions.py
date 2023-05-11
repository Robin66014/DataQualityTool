import pandas as pd
import plotly.express as px
from scipy.stats import shapiro, anderson, kstest, normaltest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import shapiro, anderson, kstest, normaltest
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import missingno as msno
from sklearn.preprocessing import LabelEncoder
import pandas_dq
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders.target_encoder import TargetEncoder


def test_normality(dataset, column_types):
    """
    Performs normality tests on all numeric columns of a dataset and returns the test results.

    Args:
        dataset (pandas DataFrame): The input dataset to test.
        column_types (list): A list of column data types for the dataset.

    Returns:
        pandas DataFrame: A dataframe with the test results for all numeric columns.
    """
    #create empty df to store the test results
    test_results = pd.DataFrame(columns=['column', 'shapiro_wilk_stat', 'shapiro_wilk_pvalue',
                                         'anderson_stat', 'anderson_crit_vals', 'anderson_sig_levels',
                                         'kolmogorov_smirnov_stat', 'kolmogorov_smirnov_pvalue',
                                         'd_agostino_pearson_stat', 'd_agostino_pearson_pvalue'])

    #iterate through each column in the dataset
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
    #TODO: waarschuwing geven als dimensionality te groot (te veel features / te weinig instances heeft)
    return None


def pandas_dq_report(dataset, target):


    #TODO fixen voor regression

    if target != 'None':
        #label encode target if categorical
        le = LabelEncoder()
        le.fit(dataset[target])
        #transform target column using the fitted encoder
        dataset[target] = le.transform(dataset[target])
        #create dq report
        report = pandas_dq.dq_report(dataset, target=target, csv_engine="pandas", verbose=1)
    else:
        report = pandas_dq.dq_report(dataset, target=None, csv_engine="pandas", verbose=1)
    #Convert to dict
    reportDICT = report.to_dict()
    print(reportDICT)
    #fix string issue (dtype) in pandas_dq conversion to a dictionary
    reportDICT = {k: {k2: str(v2).replace("dtype(", "dtype") for k2, v2 in v.items()} for k, v in reportDICT.items()}

    #make df and append list of column names to beginning of df
    reportDF = pd.DataFrame(reportDICT)
    reportDF.insert(0, 'Column', list(dataset.columns))
    #TODO: aanpassingen maken aan het report zoals: outliers zijn anders, fairness checks toevoegen
    #TODO: additional remarks; total outliers based on all column values, fairness warnings, few instances compared to amount of columns

    return reportDF


def simple_model_performance():

    #TODO: wat simpele scikit learn modellen runnen na ze geonehotencode zijn om wat performance metrics te laten zien zoals IBM doet
    #hoe dit doen? random split maken van de dataset voor train en test?



    return None


def encode_categorical_columns(dataset, target, dtypes):
    """"Function that one-hot-encodes categorical columns and label encodes the target column. It returns the encoded dataset
    and the mapping of the original labels to the encoded labels"""
    #Find all categorical columns
    #TODO: Regression werkend krijgen (label encoding dan niet van toepassing)
    categorical_cols = [] #dataset.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in dataset.columns:
        if dtypes[col] == 'categorical' or dtypes[col] == 'boolean':
            categorical_cols.append(col)

    mapping = None
    target_is_categorical = False
    if target != 'None':
        # remove target as we want to label encode this (for classification problems)
        if target in categorical_cols:
            target_is_categorical = True
            categorical_cols.remove(target)

            # label encode target
            le = LabelEncoder()
            encoded_target = le.fit_transform(dataset[target])
            encoded_labels = le.transform(dataset[target])
            mapping = {label: value for label, value in zip(encoded_labels, dataset[target])}
            # replace target column with label encoded values
            dataset.drop(columns=[target], inplace=True)
            dataset[target] = encoded_target
    if not categorical_cols:  # then no features are categorical, and we're done
        return dataset, mapping

    #if there are categorical columns, we want to one-hot-encode them

    #encode categoricals
    encoder = OneHotEncoder(handle_unknown='ignore', max_categories=100)
    encoded_columns = encoder.fit_transform(dataset[categorical_cols])
    new_columns = pd.DataFrame(encoded_columns.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

    #add new columns to df and drop old ones
    dataset_encoded = pd.concat([dataset, new_columns], axis=1)
    dataset_encoded = dataset_encoded.drop(columns=categorical_cols)

    #reposition target column to the end of the dataframe
    if target != 'None' and target_is_categorical:
        dataset_encoded.drop(columns=[target], inplace=True)
        dataset_encoded[target] = encoded_target

    #XGBClassifier doesn't accept: [, ] or <, so loop over the columns and change the names if they contain such values
    new_col_names = {col: col.replace('<', '(smaller than)').replace('[', '(').replace(']', ')') for col in dataset_encoded.columns}
    dataset_encoded = dataset_encoded.rename(columns=new_col_names)
    #print('@@@@@@@@@@@@@@@@@@@@@', type(dataset_encoded))

    return dataset_encoded, mapping

def pcp_plot(encoded_df, target):
    #TODO: tekst/lookup table toevoegen met conversie categorische variabelen encoding als dictionary
    #TODO: clean dataset?
    if target != 'None':
        fig = px.parallel_coordinates(encoded_df, color=target)
    else:
        fig = px.parallel_coordinates(encoded_df)

    return fig

def missingno_plot(df):
    #TODO: clean dataset?
    msno_plot = msno.matrix(df)

    return msno_plot

def plot_feature_importance(df, target, dtypes):
    """"plots randomforest feature importances in a horizontal barchart, based on target encoded feature values"""
    te = TargetEncoder()
    #split target from data
    x = df.drop(columns=[target])
    y = df[target]
    # TODO: clean dataset?
    if dtypes[target] == 'boolean' or dtypes[target] == 'categorical':

        le = LabelEncoder()
        y = le.fit_transform(y)
        x = te.fit_transform(x, y)
        rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        rf.fit(x, y)
    else:
        x = te.fit_transform(x, y)
        rf = RandomForestRegressor(n_estimators=10, n_jobs=-1)
        #TODO: regression

    #train random forest classifier / regressor depending on task type
    #TODO: werkend voor regression maken (afhangend van dtypes)


    #obtain feature importances
    feature_importances = rf.feature_importances_

    #create df from them
    df_feature_importances = pd.DataFrame({
        'feature': x.columns,
        'importance': feature_importances
    })

    #horizontal barchart
    fig = px.bar(df_feature_importances.sort_values('importance', ascending=False),
                 x='importance', y='feature', orientation='h',
                 title='Feature Importance')
    return fig


def plot_dataset_distributions(data, dtypes):
    """"plots the distributions per column in the dataset"""
    #TODO: functie klopt not niet helemaal, zie target iris.csv


    figs = []

    # each datatypre requires different plotting
    for col in data.columns:
        # histogram for numerical values
        if dtypes[col] == 'floating' or dtypes[col] == 'numeric' or dtypes[col] == 'integer':
            fig = px.histogram(data, x=col, title=f'{col} distribution')
            figs.append(fig)

        #bar chart for categoricals & booleans
        elif dtypes[col] == 'boolean':
            fig = px.bar(data, x=col, color=col, title=f'{col} distribution')
            figs.append(fig)
        elif dtypes[col] == 'categorical':
            fig = px.bar(data, x=col, title=f'{col} distribution')
            figs.append(fig)

        #for sentences, plot the word frequency
        elif dtypes[col] == 'sentence':
            word_counts = data[col].str.split(expand=True).stack().value_counts()
            fig = px.bar(x=word_counts.index, y=word_counts.values, title=f'{col} word cloud')
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
            figs.append(fig)

    return figs
