import pandas
import streamlit as st
from deepchecks.tabular import Dataset
import deepchecks.tabular.checks
import pandas as pd
from sklearn.datasets import load_iris
from deepchecks.tabular.datasets.classification.phishing import load_data
import numpy as np

amount_of_columns = 999999999
amount_of_samples = 999999999

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







# data = {'col1': [pd.NA, pd.NaT], 'col2': ['test', pd.NaT], 'col3': ['1', 'cat']}
# dataframe = pd.DataFrame(data=data)
# res = missing_values(dataframe)
# # print(res.to_string())
# # res = pandas.NA == 'na'
# print(type(res))

# df = pd.DataFrame({
#     'A': ['foo', 'bar', 'baz', 'foo', 'bar', 'foo', 'bar', 'baz', 'foo', 'bar', 'test', 'test'],
#     'B': [1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 99, 99],
#     'C': ['A', 'B', 'C', 'A', 'B', 'A', 'B', 'C', 'A', 'B', 'test', 'test']
# })

# data = {'Name': ['John', 'Jane', 'Bob', 'Mary', 'Kate'],
#         'Age': [25, 31, 19, 27, 22],
#         'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Boston']}

# df = pd.read_csv('datasets\iris_with_missing.csv')
# # df = pd.DataFrame(data)
# #
# res = missing_values(df)
# print(res)
