import pandas
import streamlit as st
from deepchecks.tabular import Dataset
import deepchecks.tabular.checks
import pandas as pd
from sklearn.datasets import load_iris
from deepchecks.tabular.datasets.classification.phishing import load_data
import numpy as np
#data = pd.read_csv('datasets\Iris.csv')
#dataset = Dataset(data, label = 'Species')
amount_of_columns = 999999999
amount_of_samples = 999999999

def missing_values(dataset):
    """"Checks the amount of missing values, and the types of missing values: numpy.NaN, None, '', ..."""
    checkPercentOfNulls = deepchecks.tabular.checks.PercentOfNulls(n_samples = amount_of_samples) #percentage of NA ('np.nan', 'None'...)
    resultPercentOfNulls = checkPercentOfNulls.run(dataset)
    checkMixedNulls = deepchecks.tabular.checks.MixedNulls(n_samples = amount_of_samples)
    resultMixedNulls = checkMixedNulls.run(dataset) #dictionary containing the types of potential missing values like 'N/A', ' '

    # obtain all the different types of potential missing values in the dataset
    types_of_missing_values_list  = []
    for key, value in resultMixedNulls.value.items():
        for key2, value2 in value.items():
            if key2 not in types_of_missing_values_list:
                types_of_missing_values_list.append(key2)

    #if column contains the missing value type, add it's count and percentage missing in the dictionary, else add 0
    myDict = {key: [] for key in types_of_missing_values_list}
    total_missing_in_column = []
    for key, value in resultMixedNulls.value.items():
        missing_in_column = 0
        for type_of_missing_value in types_of_missing_values_list:
            if type_of_missing_value in value:
                myDict[type_of_missing_value].append(value[type_of_missing_value])
                missing_in_column += value[type_of_missing_value]['percent']
            else:
                myDict[type_of_missing_value].append(0)
        total_missing_in_column.append(missing_in_column)


    missing_values_df = resultPercentOfNulls.value
    #TODO: potentieel extra kolommen expandable maken + totale getal missing values kloppend maken
    for key, value in myDict.items():
        missing_values_df[key] = value
    #append column to df with total missingness per column
    missing_values_df['Potential total missingness percentage in column'] = total_missing_in_column
    #TODO: uitleg dat line hierboven ook missingness in de vorm van '' meeneemt buiten NA waardes als numpy.NaN
    #TODO: n_top_columns variabel maken (door gebruiker aanpasbaar, of als het meer dan 20 kolommen zijn dan maar 10 showen oid)
    missing_values_df.rename(columns={"Percent of nulls in sample": "Percent missing (NA)"}, inplace=True)

    #if there are no 'potential' missing values, the column isn't necessary
    if missing_values_df['Percent missing (NA)'].equals(missing_values_df['Potential total missingness percentage in column']):
        missing_values_df = missing_values_df.drop('Potential total missingness percentage in column', axis=1)

    #missing_values_df.columns = missing_values_df.columns.map(str)
    missing_values_df = missing_values_df.rename(columns={'pandas.NaT': 'pandas NaT', 'pandas.NA': 'pandas NA'})
    column_names = missing_values_df.index
    missing_values_df.insert(0, 'Columns', column_names)

    return missing_values_df #returns dataframe with missing values


def duplicates(dataset):
    """"Checks whether any duplicate rows exist and displays the row numbers that are duplicate in a table"""
    checkDataDuplicates = deepchecks.tabular.checks.DataDuplicates(n_to_show=amount_of_columns, n_samples=amount_of_samples)
    resultDataDuplicates = checkDataDuplicates.run(dataset)

    #TODO: finish when result.value is fixed for this specific check
    return None








# data = {'col1': [pd.NA, pd.NaT], 'col2': ['test', pd.NaT], 'col3': ['1', 'cat']}
# dataframe = pd.DataFrame(data=data)
# res = missing_values(dataframe)
# print(res.to_string())
# res = pandas.NA == 'na'
# print(res)
