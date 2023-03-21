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
    fig = px.imshow(correlationDF) #plotly image for in Dash application

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

def outlier_detection(dataset, nearest_neighors_percent = 0.01, threshold = 0.8):
    """"Function that checks for outliers samples (jointly across all features) using
     the LoOP algorithm: (https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf)"""

    checkOutlier = deepchecks.tabular.checks.OutlierSampleDetection(nearest_neighbors_percent=nearest_neighors_percent,
                                                                         n_samples=amount_of_samples, timeout = 20, n_to_show = amount_of_samples) #TODO: compute snelheid bekijken, anders samplen zoals op deepchecks
    #TODO: timeout warning toevoegen, nu geeft ie gewoon een error
    resultOutlier = checkOutlier.run(dataset)
    result = resultOutlier.display[1] #obtain dataframe with probability scores
    #TODO: filteren van dataframe loskoppelen van functie, anders moet het steeds herberekend worden als de callback aan deze functie wordt gekoppeld
    result_filtered = result[result['Outlier Probability Score'] > threshold] #obtain only the outliers that have a probability higher than the desired threshold
    amount_of_outliers = result_filtered.shape[0]

    return result_filtered, amount_of_outliers


## for testing purposes
#data = {'col1': [pd.NA, pd.NaT], 'col2': ['test', pd.NaT], 'col3': ['1', 'cat']}
# dataframe = pd.DataFrame({
#     'a': ['Deep', np.nan, 'deep', 'deep!'],
#     'b': [2, 3, 4, 8],
#     'c': [None, 'weeehooo', 'weeehoo', 'Weeehooo'],
#     'd': ['a', 4, 'ploep', 'hoi'],
# })
# ds = adult.load_data(as_train_test=False)
# outlier_detection(ds)