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
import plotly.graph_objects as go
#dataset = Dataset(data, label = 'Species')
amount_of_columns = 10000000
amount_of_samples = 10000000 #TODO: compute snelheid bekijken, anders samplen zoals op deepchecks


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
    correlationDF.rename_axis('Column', inplace=True)
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
                                                                         n_samples=10000, timeout = 1200, n_to_show = amount_of_samples) #TODO: compute snelheid bekijken, anders samplen zoals op deepchecks
        #TODO: timeout warning toevoegen, nu geeft ie gewoon een error
        resultOutlier = checkOutlier.run(dataset)
        result = resultOutlier.display[1] #obtain dataframe with probability scores
        row_numbers = result.index
        result.insert(0, 'Row number', row_numbers)
        result.sort_values('Row number', inplace=True)
        outlier_prob_scores = result['Outlier Probability Score']
        max_prob_score = result['Outlier Probability Score'].max()

        #TODO: filteren van dataframe loskoppelen van functie, anders moet het steeds herberekend worden als de callback aan deze functie wordt gekoppeld
        result_filtered = result[result['Outlier Probability Score'] > threshold] #obtain only the outliers that have a probability higher than the desired threshold

        amount_of_outliers = 0
        if result_filtered.empty:
            result_filtered = pd.DataFrame({"Check notification": ["Check passed: No outliers with a probability score higher than {}, The highest probability found is: {}".format(threshold, max_prob_score)]})
            dq_issue_report_string = ' '
        else:
            amount_of_outliers = result_filtered.shape[0]
            dq_issue_report_string = f'{amount_of_outliers} outlier instances encountered, check their legitimacy and or remove them.'
        result_filtered = dash_datatable_format_fix(result_filtered)

        return result_filtered, amount_of_outliers, threshold, outlier_prob_scores, dq_issue_report_string

    except Exception as e:
        return pd.DataFrame({"Check notification": ["COMPUTATION TOO EXPENSIVE ERROR: MAXIMUM COMPUTATION TIME OF 20 MINUTES EXCEEDED. Note: if your dataset is very small, this could also be the cause of the check not running."]}), 0, threshold, pd.DataFrame(), ' '

def box_plot(df, dtypes):

    numerical_columns = []
    for col in df.columns:
        # histogram for numerical values
        if dtypes[col] == 'floating' or dtypes[col] == 'integer' or dtypes[col] == 'numeric':
            numerical_columns.append(col)
            # Add traces for each numerical column
    data = []
    for column in numerical_columns:
        data.append(go.Box(y=df[column], name=column))

    layout = go.Layout(
    title="Numerical column outlier Visualization",
    yaxis_title="Value"
                        )
    fig = go.Figure(data=data, layout=layout)


    return fig


