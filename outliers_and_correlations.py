import pandas
from deepchecks.tabular import Dataset
import deepchecks.tabular.checks
import pandas as pd
from plot_and_transform_functions import dash_datatable_format_fix
from sklearn.datasets import load_iris
from deepchecks.tabular.datasets.classification.phishing import load_data
import numpy as np
from deepchecks.tabular.datasets.classification import adult
import plotly.express as px
import plotly.figure_factory as ff
#dataset = Dataset(data, label = 'Species')
amount_of_columns = 100000
amount_of_samples = 100000 #TODO: compute snelheid bekijken, anders samplen zoals op deepchecks


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
    correlationDF = dash_datatable_format_fix(correlationDF)
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

    correlationDF = dash_datatable_format_fix(correlationDF)
    return correlationDF


def outlier_detection(dataset, nearest_neighors_percent = 0.01, threshold = 0.80):
    """"Function that checks for outliers samples (jointly across all features) using
     the LoOP algorithm: (https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf)"""
    #TODO: outliers ook zichtbaar maken in overeenstemming met dq_report / op basis van minimum value
    try:
        checkOutlier = deepchecks.tabular.checks.OutlierSampleDetection(nearest_neighbors_percent=nearest_neighors_percent,
                                                                         n_samples=10000, timeout = 1000, n_to_show = amount_of_samples) #TODO: compute snelheid bekijken, anders samplen zoals op deepchecks
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
        result_filtered = dash_datatable_format_fix(result_filtered)

        return result_filtered, amount_of_outliers, threshold

    except Exception as e:
        return pd.DataFrame({"COMPUTATION TOO EXPENSIVE ERROR: MAXIMUM COMPUTATION TIME EXCEEDED"}), 0, 0





## for testing purposes
#data = {'col1': [pd.NA, pd.NaT], 'col2': ['test', pd.NaT], 'col3': ['1', 'cat']}
# dataframe = pd.DataFrame({
#     'a': ['Deep', np.nan, 'deep', 'deep!'],
#     'b': [2, 3, 4, 8],
#     'c': [None, 'weeehooo', 'weeehoo', 'Weeehooo'],
#     'd': ['a', 4, 'ploep', 'hoi'],
# })
# zero_data = np.zeros(shape=(100,100))
# ds = pd.DataFrame(zero_data)

# ds = adult.load_data(as_train_test=False)
# checkClassImbalance = deepchecks.tabular.checks.ClassImbalance(n_top_labels=amount_of_columns)
# resultClassImbalance = checkClassImbalance.run(ds)
#
# result = resultClassImbalance.value #pandas dataframe with correlation values
# resultDF = pd.DataFrame(result, index=[0])
# print('@@ds.label_co: ', ds.label_col)
# print('@@ds.data: ', ds.data)
# print('@@ds.classes_in_label_col: ', ds.classes_in_label_col)
# print('@@ds.columns_info: ', ds.columns_info)
# print('@@ds.cat_features: ', ds.cat_features)
#

# res, amount = outlier_detection(ds)
# print(res)
# print(res.iloc[0,:])
# print(amount)