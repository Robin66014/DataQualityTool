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
#dataset = Dataset(data, label = 'Species')


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
    resultDF = resultConflictingLabels.display[1]

    return resultDF


def wrong_label(dataset):
    """"Function that finds potential label errors (due to annotator mistakes), edge cases, and otherwise ambiguous examples"""
    #cleanlab + deepchecks
    import cleanlab
    from cleanlab.classification import CleanLearning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder


    #labels = [np.where(np.array(list(dict.fromkeys(labels))) == e)[0][0] for e in labels]

    # dataset_no_label = dataset.features_columns
    # labels = dataset.label_col.tolist()
    # encoded_labels = LabelEncoder().fit_transform(labels)
    # print('dataset: ', dataset_no_label)
    # print('encoded_labels: ', encoded_labels)
    # yourFavoriteModel = LogisticRegression(verbose=0, random_state=0)
    # issues = CleanLearning(yourFavoriteModel, seed=0).find_label_issues(X= dataset_no_label, labels = encoded_labels)
    # print(issues.head())

    return None



from deepchecks.tabular.datasets.classification import adult
ds = adult.load_data(as_train_test= False)

wrong_label(ds)