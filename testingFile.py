import pandas
from deepchecks.tabular import Dataset
import deepchecks.tabular.checks
import pandas as pd
from sklearn.datasets import load_iris
from deepchecks.tabular.datasets.classification.phishing import load_data
import numpy as np

#TODO: start
def amount_of_diff_values(dataset):
    """"displays the amount of different values per column"""
    #TODO: currently only displays the amount of distinct values in a column, do something with the case that it is literally unique
    checkSingleValue = deepchecks.tabular.checks.IsSingleValue(n_to_show=amount_of_columns, n_samples=amount_of_samples)
    resultSingleValue = checkSingleValue.run(dataset)
    result_dict = resultSingleValue.value

    for key, value in result_dict.items():
        result_dict[key] = [value] #contains the amount of distinct values per column


    df = pd.DataFrame.from_dict(result_dict, orient = 'columns')

    return df


def mixed_data_types(dataset):
    """"function that discovers the various types of data that exist in a column"""
    #TODO: nog goed controleren op mogelijke edge cases, mogelijk foutgevoelig (strings)?
    checkMixedDataTypes = deepchecks.tabular.checks.MixedDataTypes(n_top_columns=amount_of_columns, n_samples=amount_of_samples)
    resultMixedDataTypes = checkMixedDataTypes.run(dataset)


    result_dict = resultMixedDataTypes.value

    #Deepchecks gives no values when the column is 100% string or number, so we create it manually
    for key in result_dict:
        if not bool(result_dict[key]): #then the value is {} (empty)
            random_samples = list(dataset[key].sample(n=3)) #obtain 3 random samples
            if pd.api.types.is_string_dtype(dataset[key]): #check whether the values in the column are strings
                result_dict[key] = {'strings': 1.0, 'numbers': 0.0, 'strings_examples': {', '.join(map(str, random_samples))}, 'numbers_examples': {}}
            else:
                result_dict[key] = {'strings': 0.0, 'numbers': 1.0, 'strings_examples': {}, 'numbers_examples': {', '.join(map(str, random_samples))}}

    df = pd.DataFrame.from_dict(result_dict, orient='columns')
    index_names = df.index
    df.insert(0, 'Data type', index_names)
    #df = pd.DataFrame(resultMixedDataTypes.display[1])

    return df


def special_characters(dataset):
    """"function that checks whether values exist in the column that contain only special characters like #, ?, -, if so displays
     the column with the 5 most common special characters
    """
    checkSpecialCharacters = deepchecks.tabular.checks.SpecialCharacters(n_top_columns=amount_of_columns,
                                                                   n_samples=amount_of_samples, n_most_common = 5)
    resultSpecialCharacters = checkSpecialCharacters.run(dataset)

    result = resultSpecialCharacters.display
    if result:
        df = pd.DataFrame(result[1])
        column_names = df.index
        df.insert(0, 'Column', column_names)
    else:
        df = pd.DataFrame({"Message": ["No special characters encountered"]})
    return df


def string_mismatch(dataset):
    """"Function that checks the cell entity, e.g 'red' and 'Red' and 'red!' are probably meant to be the same values"""
    checkStringMismatch = deepchecks.tabular.checks.StringMismatch(n_top_columns=amount_of_columns,
                                                                         n_samples=amount_of_samples)
    resultStringMismatch = checkStringMismatch.run(dataset)


    result = resultStringMismatch.display
    if result:
        df = pd.DataFrame(result[1])
        df = pd.DataFrame(df.to_records()) #flatten hierarchical index in columns
    else:
        df = pd.DataFrame({"Message": ["No string mismatch or variants of the same string encountered"]})

    return df




#TODO: (als je dit nog wilt) String length out of bounds


import pandas

from deepchecks.tabular import Dataset
import deepchecks.tabular.checks
import pandas as pd
from sklearn.datasets import load_iris
from deepchecks.tabular.datasets.classification.phishing import load_data
import numpy as np


def missing_values(dataset):
    """"Checks the amount of missing values, and the types of missing values: numpy.NaN, None, '', ..."""
    checkPercentOfNulls = deepchecks.tabular.checks.PercentOfNulls(n_samples = amount_of_samples) #percentage of NA ('np.nan', 'None'...)
    resultPercentOfNulls = checkPercentOfNulls.run(dataset)
    checkMixedNulls = deepchecks.tabular.checks.MixedNulls(n_samples = amount_of_samples)
    resultMixedNulls = checkMixedNulls.run(dataset) #dictionary containing the types of potential missing values like 'N/A', ' '

    # create empty dictionary to store the counts of zeros
    zeros = []
    zeros_percentage_list = []
    for col in dataset.columns:
        # count the number of zeros in the column and add to dictionary
        zeros_count = (dataset[col] == 0).sum()
        zeros_percentage = (dataset[col] == 0).mean() * 100
        zeros_percentage = round(zeros_percentage, 2)
        zeros.append(zeros_count)
        zeros_percentage_list.append(zeros_percentage)
    # create a new dataframe from the zeros_dict

    # obtain all the different types of potential missing values in the dataset
    types_of_missing_values_list  = []
    for key, value in resultMixedNulls.value.items():
        for key2, value2 in value.items():
            if key2 not in types_of_missing_values_list:
                types_of_missing_values_list.append(key2)

    #if column contains the missing value type, add it's count and percentage missing in the dictionary, else add 0
    myDict = {key: [] for key in types_of_missing_values_list}
    total_missing_in_column = []
    #print('@@resultMixedNulls.value.items', resultMixedNulls.value.items())
    for key, value in resultMixedNulls.value.items():
        missing_in_column = 0
        for type_of_missing_value in types_of_missing_values_list:
            if type_of_missing_value in value:
                count = value[type_of_missing_value]['count']
                percent = value[type_of_missing_value]['percent'] * 100
                formatted_string = "{} ({:.3f}%)".format(count, percent)
                myDict[type_of_missing_value].append(formatted_string)
                missing_in_column += percent
            else:
                myDict[type_of_missing_value].append(0)
        total_missing_in_column.append(missing_in_column)

    missing_values_df = resultPercentOfNulls.value

    for key, value in myDict.items():
        missing_values_df[key] = value

    #append column to df with total missingness per column
    missing_values_df['Potential total missingness percentage in column'] = total_missing_in_column
    missing_values_df.rename(columns={"Percent of nulls in sample": "Percent missing (NA)"}, inplace=True)

    column_names = missing_values_df.index
    missing_values_df.insert(0, 'Columns', column_names)

    modified_list = [name.replace('"', '') for name in missing_values_df.columns]  # remove double quotes in values, like '"None"' --> 'None'
    missing_values_df.columns = modified_list
    # append the zeros list to the original dataframe
    missing_values_df.insert(len(missing_values_df.columns) - 1, 'Zeros', zeros)
    missing_values_df['Potential total missingness percentage in column'] = [x + zeros_percentage_list[i] for i, x in enumerate(missing_values_df['Potential total missingness percentage in column'])]

    return missing_values_df #returns dataframe with missing values

def duplicates(df):
    """"Checks whether any duplicate rows exist and displays the row numbers that are duplicate in a table"""
    # checkDataDuplicates = deepchecks.tabular.checks.DataDuplicates(n_to_show=amount_of_columns, n_samples=amount_of_samples)
    # resultDataDuplicates = checkDataDuplicates.run(dataset)

    try:
        #Check for duplicate rows find list of row numbers with the same feature values
        duplicates = df[df.duplicated(keep=False)]
        grouped = duplicates.groupby(list(df.columns), group_keys=False).apply(lambda x: list(x.index)).reset_index(name='Duplicates')

        #merge & add original row number
        merged = pd.merge(df.reset_index(), grouped, on=list(df.columns), how='left')
        merged['Duplicates'] = merged.apply(lambda x: sorted(list(set(x['Duplicates'] + [x['index']]))), axis=1)
        merged = merged.rename(columns={'index':'Index first encountered'})
        #Drop duplicates from the df
        final_df = merged.drop_duplicates(subset=list(df.columns))

        return final_df

    except Exception as e:
        return pd.DataFrame({"Message": ["No duplicates encountered"]})



def duplicate_column():


    return None


import pandas
from deepchecks.tabular import Dataset
import deepchecks.tabular.checks
import pandas as pd
from sklearn.datasets import load_iris
from deepchecks.tabular.datasets.classification.phishing import load_data
import numpy as np
from deepchecks.tabular.datasets.classification import adult
import plotly.express as px
import plotly.figure_factory as ff



def feature_feature_correlation(dataset):
    """"computes the correlation between each feature pair;
    Methods to calculate for each feature label pair:

    numerical-numerical: Pearson’s correlation coefficient
    numerical-categorical: Correlation ratio
    categorical-categorical: Symmetric Theil’s U
    """

    checkFeatureFeatureCorrelation = deepchecks.tabular.checks.FeatureFeatureCorrelation(n_top_columns=amount_of_columns,
                                                                         n_samples=amount_of_samples) #TODO: compute snelheid bekijken, anders samplen zoals op deepchecks
    resultFeatureFeatureCorrelation = checkFeatureFeatureCorrelation.run(dataset)

    correlationDF = resultFeatureFeatureCorrelation.value #pandas dataframe with correlation values
    fig = px.imshow(correlationDF, text_auto=True, aspect="auto", color_continuous_scale='thermal') #plotly image for in Dash application
    #fig = ff.create_annotated_heatmap(correlationDF)
    return correlationDF, fig

def feature_label_correlation(dataset):
    """"computes the correlation between each feature and the label;
    Methods to calculate for each feature label pair:

    numerical-numerical: Pearson’s correlation coefficient
    numerical-categorical: Correlation ratio
    categorical-categorical: Symmetric Theil’s U
    """

    checkFeatureLabelCorrelation = deepchecks.tabular.checks.FeatureLabelCorrelation(n_top_columns=amount_of_columns,
                                                                         n_samples=amount_of_samples) #TODO: compute snelheid bekijken, anders samplen zoals op deepchecks
    resultFeatureLabelCorrelation = checkFeatureLabelCorrelation.run(dataset)

    result_dict = resultFeatureLabelCorrelation.value

    #convert to desired format and round to 3 decimals
    correlationDF = pd.DataFrame(result_dict, index=[0]).round(3)


    return correlationDF

def identifier_label_correlation(dataset):
    #TODO: MAKEN INDIEN NODIG

    return None

def feature_importance(dataset):
    #TODO: maken met random forest
    return None

def outlier_detection(dataset, nearest_neighors_percent = 0.01, threshold = 0.80):
    """"Function that checks for outliers samples (jointly across all features) using
     the LoOP algorithm: (https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf)"""
    #TODO: outliers ook zichtbaar maken in overeenstemming met dq_report / op basis van minimum value
    try:
        checkOutlier = deepchecks.tabular.checks.OutlierSampleDetection(nearest_neighbors_percent=nearest_neighors_percent,
                                                                         n_samples=10000, timeout = 20, n_to_show = amount_of_samples) #TODO: compute snelheid bekijken, anders samplen zoals op deepchecks
        #TODO: timeout warning toevoegen, nu geeft ie gewoon een error
        resultOutlier = checkOutlier.run(dataset)
        result = resultOutlier.display[1] #obtain dataframe with probability scores
        row_numbers = result.index
        result.insert(0, 'Row number', row_numbers)
        max_prob_score = result['Outlier Probability Score'].max()

        #TODO: filteren van dataframe loskoppelen van functie, anders moet het steeds herberekend worden als de callback aan deze functie wordt gekoppeld
        result_filtered = result[result['Outlier Probability Score'] > threshold] #obtain only the outliers that have a probability higher than the desired threshold

        amount_of_outliers = 0
        if result_filtered.empty:
            result_filtered = pd.DataFrame({"Message": ["No outliers with a probability score higher than {}, The highest probability found is: {}".format(threshold, max_prob_score)]})
        else:
            amount_of_outliers = result_filtered.shape[0]

        return result_filtered, amount_of_outliers, threshold

    except Exception as e:
        return pd.DataFrame({"COMPUTATION TOO EXPENSIVE ERROR": [e]}), None



import pandas
from deepchecks.tabular import Dataset
import deepchecks.tabular.checks
import pandas as pd
from sklearn.datasets import load_iris
from deepchecks.tabular.datasets.classification.phishing import load_data
import numpy as np
from deepchecks.tabular.datasets.classification import adult
import plotly.express as px
import cleanlab
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

#TODO: END
def class_imbalance(dataset):
    """"Function that checks the distribution of the target variable / label"""

    #TODO: rekening houden dat deze test irrelevant is voor sommige (regression) problems
    checkClassImbalance = deepchecks.tabular.checks.ClassImbalance(n_top_labels=amount_of_columns)
    resultClassImbalance = checkClassImbalance.run(dataset)

    result = resultClassImbalance.value #pandas dataframe with correlation values
    resultDF = pd.DataFrame(result, index=[0])
    # fig = px.bar(resultDF) #plotly image for in Dash application
    # fig.show()
    return resultDF#, fig


def conflicting_labels(dataset):
    """"Function that checks for datapoints with exactly the same feature values, but different labels"""

    checkConflictingLabels = deepchecks.tabular.checks.ConflictingLabels(n_to_show=20)
    resultConflictingLabels = checkConflictingLabels.run(dataset)

    #percentage = round(resultConflictingLabels.value[0], 3) #pandas dataframe with correlation values
    result = resultConflictingLabels.value
    percentage = round(result.get('percent'), 6)
    if len(result['samples_indices']) == 0:
        resultDF = pd.DataFrame({"Message": ["No conflicting labels encountered"]})
    else:
        resultDF = resultConflictingLabels.display[1]

    return resultDF, percentage


def wrong_label(encoded_dataset, target):
    """"Function that finds potential label errors (due to annotator mistakes), edge cases, and otherwise ambiguous examples"""
    model_XGBC = XGBClassifier(tree_method="hist", enable_categorical=True)  # hist is fastest tree method of XGBoost, use default model
    # TODO: def main problem? testen

    data_no_labels = encoded_dataset.drop(columns=[target])
    labels = encoded_dataset[target]

    #obtain predicted probabilities using 5 fold cross validation
    pred_probs = cross_val_predict(model_XGBC, data_no_labels, labels, method='predict_proba')

    preds = np.argmax(pred_probs, axis=1)
    acc_original = accuracy_score(preds, labels)
    print(f"Accuracy with original data: {round(acc_original * 100, 1)}%")

    #use cleanlabs built in confident learning method to find label issues
    cl = cleanlab.classification.CleanLearning()
    issues_dataframe = cl.find_label_issues(X=None, labels=labels, pred_probs=pred_probs)
    wrong_label_count = (issues_dataframe['is_label_issue'] == True).sum()

    # filter df so only errors are visible
    issues_dataframe_only_errors = issues_dataframe[issues_dataframe['is_label_issue'] == True]

    return issues_dataframe, issues_dataframe_only_errors, wrong_label_count



#TODO: een functie maken waarin ik de class parity check (prediction accuracy per target group)

from deepchecks.tabular.datasets.classification import adult

import pandas as pd
import plotly.express as px
from scipy.stats import shapiro, anderson, kstest, normaltest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import shapiro, anderson, kstest, normaltest
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import missingno as msno
from sklearn.preprocessing import LabelEncoder
import pandas_dq
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
        if dtypes[col] == 'categorical':
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

    if target != 'None':
        fig = px.parallel_coordinates(encoded_df, color=target)
    else:
        fig = px.parallel_coordinates(encoded_df)

    return fig

def missingno_plot(df):

    msno_plot = msno.matrix(df)

    return msno_plot

def plot_feature_importance(df, target):
    """"plots randomforest feature importances in a horizontal barchart, based on target encoded feature values"""
    te = TargetEncoder()
    #split target from data
    x = df.drop(columns=[target])
    y = df[target]
    x = te.fit_transform(x, y)
    #train random forest classifier / regressor depending on task type
    #TODO: werkend voor regression maken (afhangend van dtypes)
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(x, y)

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

import pandas as pd
import sortinghatinf #algorithm to predict the feature type out og [numeric, categorical, datetime, sentence,
from deepchecks.tabular import Dataset
from deepchecks.tabular.datasets.classification import adult
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
