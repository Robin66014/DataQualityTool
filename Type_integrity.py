import pandas
import streamlit as st
from deepchecks.tabular import Dataset
import deepchecks.tabular.checks
import pandas as pd
from sklearn.datasets import load_iris
from deepchecks.tabular.datasets.classification.phishing import load_data
import numpy as np
data = pd.read_csv('datasets\Iris.csv')
#dataset = Dataset(data, label = 'Species')
amount_of_columns = 999999999
amount_of_samples = 999999999


def single_value(dataset):
    """"Checks whether redundant columns exist that contain one single unique value, and displays this value"""
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
    df = pd.DataFrame(result[1])

    return df


def string_mismatch(dataset):
    """"Function that checks the cell entity, e.g 'red' and 'Red' and 'red!' are probably meant to be the same values"""
    checkStringMismatch = deepchecks.tabular.checks.StringMismatch(n_top_columns=amount_of_columns,
                                                                         n_samples=amount_of_samples)
    resultStringMismatch = checkStringMismatch.run(dataset)

    result = resultStringMismatch.display[1]
    df = pd.DataFrame(result)

    return df


#TODO: (als je dit nog wilt) String length out of bounds



#data = {'col1': [pd.NA, pd.NaT], 'col2': ['test', pd.NaT], 'col3': ['1', 'cat']}
dataframe = pd.DataFrame({
    'a': ['Deep', np.nan, 'deep', 'deep!'],
    'b': [2, 3, 4, 8],
    'c': [None, 'weeehooo', 'weeehoo', 'Weeehooo'],
    'd': ['a', 4, 'ploep', 'hoi'],
})
string_mismatch(dataframe)
