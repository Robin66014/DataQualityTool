import deepchecks.tabular.checks
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import plotly.express as px
import cleanlab
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from plot_and_transform_functions import dash_datatable_format_fix

amount_of_columns = 10000000
amount_of_samples = 10000000


def class_imbalance(dataset):
    """"Check 5: Function that checks the distribution of the target variable / label and reports if imbalance is extremely high"""
    #deepchecks imbalance check
    checkClassImbalance = deepchecks.tabular.checks.ClassImbalance(n_top_labels=amount_of_columns)
    resultClassImbalance = checkClassImbalance.run(dataset)

    result = resultClassImbalance.value #pandas dataframe with correlation values
    fig = px.bar(x=list(result.keys()), y=list(result.values()), text=list(result.values()))
    resultDF = pd.DataFrame(result, index=[0])
    #convert to desired dash format
    resultDF = dash_datatable_format_fix(resultDF)
    #convert deepchecks result to float
    max_value = float(resultDF.max().max())
    min_value = float(resultDF.min().min())
    ratio_min_max = min_value / max_value
    ratio_min_max = round((min_value / max_value), 2)
    #if ratio is less than 1 to 100 between most infrequent and most frequent class, report this as a potential problem
    if ratio_min_max <= 0.01:
        dq_issue_report_string = f'Highly imbalanced classes with ratio {ratio_min_max}, check whether this forms a problem in your context' \
                                 f' and/or fix by resampling, cost-sensitive learning or ensemble methods.'
    else:
        dq_issue_report_string = ' '

    return resultDF, fig, dq_issue_report_string


def conflicting_labels(dataset):
    """"Check 6: Conflicting labels. Function that checks for datapoints with exactly the same feature values, but different labels"""
    #deepchecks result
    checkConflictingLabels = deepchecks.tabular.checks.ConflictingLabels(n_to_show=20)
    resultConflictingLabels = checkConflictingLabels.run(dataset)

    result = resultConflictingLabels.value
    percentage = round(result.get('percent'), 6)*100
    if len(result['samples_indices']) == 0:
        #df placeholder if check is passed
        resultDF = pd.DataFrame({"Check notification": ["Check passed: No conflicting labels encountered"]})
        dq_issue_report_string = ' '
    else:
        resultDF = resultConflictingLabels.display[1]
        resultDF.reset_index(inplace=True)
        columns = resultDF.columns.tolist()
        columns[2] = 'Feature values'
        resultDF.columns = columns
        new_df = pd.DataFrame()
        new_df['Conflicting'] = resultDF['Instances'].str.count(',') + 1
        total_conflicting = int(new_df['Conflicting'].sum())
        dq_issue_report_string = f'{total_conflicting} conflicting labels encountered, check their legimitacy and/or remove them.'
    #fix dash format issues
    resultDF = dash_datatable_format_fix(resultDF)

    return resultDF, percentage, dq_issue_report_string


def cleanlab_label_error(encoded_dataset, target):
    """"Check 7: Conflicting labels. Function that finds potential label errors (due to annotator mistakes),
    edge cases, and otherwise ambiguous examples"""
    #XGB gradient boosting ensemble

    model_XGBC = XGBClassifier(tree_method="hist", enable_categorical=True)  # hist is fastest tree method of XGBoost, use default model
    #prevent errors with object columns
    object_columns = encoded_dataset.select_dtypes(include=['object']).columns
    encoded_dataset = encoded_dataset.drop(columns=object_columns)

    data_no_labels = encoded_dataset.drop(columns=[target])
    labels = encoded_dataset[target]

    #create cross validation folds KFold object
    stratified_splits = StratifiedKFold(n_splits=5)

    #otain predicted probabilities using 5-fold stratified cross-validation
    pred_probs = cross_val_predict(model_XGBC, data_no_labels, labels, cv=stratified_splits, method='predict_proba')

    preds = np.argmax(pred_probs, axis=1)
    accuracy_score_xgbc = round((accuracy_score(preds, labels) * 100), 1)

    #use cleanlabs built in confident learning method to find label issues
    cl = cleanlab.classification.CleanLearning()
    issues_dataframe = cl.find_label_issues(X=None, labels=labels, pred_probs=pred_probs)
    issues_dataframe = issues_dataframe.reset_index()
    #count total wrong labels
    wrong_label_count = (issues_dataframe['is_label_issue'] == True).sum()

    #filter df so only errors are visible
    issues_dataframe_only_errors = issues_dataframe[issues_dataframe['is_label_issue'] == True]
    issues_dataframe = dash_datatable_format_fix(issues_dataframe)
    issues_dataframe_only_errors = dash_datatable_format_fix(issues_dataframe_only_errors)

    return issues_dataframe, issues_dataframe_only_errors, wrong_label_count, accuracy_score_xgbc

