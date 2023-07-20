import deepchecks.tabular.checks
import pandas as pd
from plot_and_transform_functions import dash_datatable_format_fix
amount_of_columns = 10000000
amount_of_samples = 10000000



def amount_of_diff_values(dataset):
    """"Check 8: single value in column. Displays the amount of different values per column"""
    #obtain deepchecks check result
    checkSingleValue = deepchecks.tabular.checks.IsSingleValue(n_to_show=amount_of_columns, n_samples=amount_of_samples)
    resultSingleValue = checkSingleValue.run(dataset)
    result_dict = resultSingleValue.value

    for key, value in result_dict.items():
        result_dict[key] = [value] #contains the amount of distinct values per column

    #add to comprehensive df
    df = pd.DataFrame.from_dict(result_dict, orient = 'columns')
    df = dash_datatable_format_fix(df)

    return df


def mixed_data_types(dataset):
    """"Check 9: mixed data types. Function that discovers the various types of data that exist in a column"""
    #obtain deepchecks check result
    checkMixedDataTypes = deepchecks.tabular.checks.MixedDataTypes(n_top_columns=amount_of_columns, n_samples=amount_of_samples)
    resultMixedDataTypes = checkMixedDataTypes.run(dataset)


    result_dict = resultMixedDataTypes.value

    #Deepchecks gives no values when the column is 100% string or number, so we create it manually
    for key in result_dict:
        if not bool(result_dict[key]): #then the value is {} (empty)
            random_samples = list(dataset[key].sample(n=3)) #obtain 3 random samples
            if pd.api.types.is_string_dtype(dataset[key]): #check whether the values in the column are strings
                result_dict[key] = {'strings': 1.0, 'numbers': 0.0, 'strings_examples': random_samples, 'numbers_examples': str([])}
            else:
                result_dict[key] = {'strings': 0.0, 'numbers': 1.0, 'strings_examples': str([]), 'numbers_examples': random_samples}

    df = pd.DataFrame.from_dict(result_dict, orient='columns')

    index_names = df.index
    if 'Data type' not in df.columns:
        df.insert(0, 'Data type', index_names)

    #be more clear in that fractions are being displayed
    df.at['strings', 'Data type'] = 'strings_fraction'
    df.at['numbers', 'Data type'] = 'numbers_fraction'
    df = dash_datatable_format_fix(df)

    return df

def special_characters(dataset):
    """"Check 10: Function that checks whether values exist in the column that contain only special characters like #, ?, -, if so displays
     the column with the 5 most common special characters
    """
    #obtain deepchecks check result
    checkSpecialCharacters = deepchecks.tabular.checks.SpecialCharacters(n_top_columns=amount_of_columns,
                                                                   n_samples=amount_of_samples, n_most_common = 5)
    resultSpecialCharacters = checkSpecialCharacters.run(dataset)
    result = resultSpecialCharacters.display
    if result:
        df = pd.DataFrame(result[1])
        column_names = df.index
        df.insert(0, 'Column', column_names)
    else:
        df = pd.DataFrame({"Check notification": ["Check passed: No special characters encountered"]})

    df.reset_index(drop=True, inplace=True)
    #format fix
    df = dash_datatable_format_fix(df)

    return df


def string_mismatch(dataset):
    """"Check 11; Function that checks the cell entity, e.g 'red' and 'Red' and 'red!' are probably meant to be the same values"""
    #obtain deepchecks check result
    checkStringMismatch = deepchecks.tabular.checks.StringMismatch(n_top_columns=amount_of_columns,
                                                                         n_samples=amount_of_samples)
    resultStringMismatch = checkStringMismatch.run(dataset)

    result = resultStringMismatch.display
    if result:
        df = pd.DataFrame(result[1])
        df = pd.DataFrame(df.to_records()) #flatten hierarchical index in columns
    else:
        df = pd.DataFrame({"Check notification": ["Check passed: No string mismatch/variants of the same string encountered"]})
    #format fix
    df = dash_datatable_format_fix(df)
    return df


