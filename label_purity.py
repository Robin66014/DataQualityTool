import pandas
import streamlit as st
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

amount_of_columns = 999999999
amount_of_samples = 10000 #TODO: compute snelheid bekijken, anders samplen zoals op deepchecks


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
    # TODO: add more models and choose whichever gives highest prediction? Downside --> slows process

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
# ds = adult.load_data(as_train_test= False)
# zero_data = np.zeros(shape=(100,5))
# df = pd.DataFrame(zero_data)
# df.iloc[0,4] = 1
# df.iloc[0,3] = 1
# df.iloc[0,2] = 1
# df.iloc[0,1] = 1
# df.iloc[0,1] = 1
# ds = Dataset(df, label = 4)
#
# res = conflicting_labels(ds)
# print(res)