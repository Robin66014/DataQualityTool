import deepchecks.tabular.checks
import pandas as pd
from plot_and_transform_functions import dash_datatable_format_fix

amount_of_columns = 10000000
amount_of_samples = 10000000

def missing_values(dataset):
    """"Check 3&4: missing values: Checks the amount of missing values, and the types of missing values: numpy.NaN, None, '', ...,
    but also checks for potential missing values 0, '-' and '?'."""
    #deepcheck checks
    checkPercentOfNulls = deepchecks.tabular.checks.PercentOfNulls(n_samples = amount_of_samples) #percentage of NA ('np.nan', 'None'...)
    resultPercentOfNulls = checkPercentOfNulls.run(dataset)
    checkMixedNulls = deepchecks.tabular.checks.MixedNulls(n_samples = amount_of_samples)
    resultMixedNulls = checkMixedNulls.run(dataset) #dictionary containing the types of potential missing values like 'N/A', ' '

    zeros = []
    question_marks = []
    question_marks_percentage_list = []
    dashes = []
    dashes_percentage_list = []
    zeros_percentage_list = []
    for col in dataset.columns:
        # count the number of zeros, question marks and dashes in the column and add to dictionary
        zeros_count = (dataset[col] == 0).sum()
        question_marks_count = (dataset[col] == '?').sum()
        dashes_count = (dataset[col] == '-').sum()
        #zeros
        zeros_percentage = (dataset[col] == 0).mean() * 100
        zeros_percentage = round(zeros_percentage, 2)
        if zeros_count != 0:
            zeros_count = "{} ({:.3f}%)".format(zeros_count, zeros_percentage)
        zeros.append(zeros_count)
        zeros_percentage_list.append(zeros_percentage)
        #question marks
        question_marks_percentage = (dataset[col] == '?').mean() * 100
        question_marks_percentage = round(question_marks_percentage, 2)
        if question_marks_count != 0:
            question_marks_count = "{} ({:.3f}%)".format(question_marks_count, question_marks_percentage)
        question_marks.append(question_marks_count)
        question_marks_percentage_list.append(question_marks_percentage)
        #dashes
        dashes_percentage = (dataset[col] == '-').mean() * 100
        dashes_percentage = round(dashes_percentage, 2)
        if dashes_count != 0:
            dashes_count = "{} ({:.3f}%)".format(dashes_count, dashes_percentage)
        dashes.append(dashes_count)
        dashes_percentage_list.append(dashes_percentage)

    # obtain all the different types of potential missing values in the dataset using deepchecks
    types_of_missing_values_list  = []
    for key, value in resultMixedNulls.value.items():
        for key2, value2 in value.items():
            if key2 not in types_of_missing_values_list:
                types_of_missing_values_list.append(key2)

    #if column contains the missing value type, add it's count and percentage missing in the dictionary, else add 0 (for nice displaying)
    missing_values_dict = {key: [] for key in types_of_missing_values_list}
    total_missing_in_column = []
    for key, value in resultMixedNulls.value.items():
        missing_in_column = 0
        for type_of_missing_value in types_of_missing_values_list:
            if type_of_missing_value in value:
                count = value[type_of_missing_value]['count']
                percent = value[type_of_missing_value]['percent'] * 100
                formatted_string = "{} ({:.3f}%)".format(count, percent)
                missing_values_dict[type_of_missing_value].append(formatted_string)
                missing_in_column += percent
            else:
                missing_values_dict[type_of_missing_value].append(0)
        total_missing_in_column.append(missing_in_column)

    missing_values_df = resultPercentOfNulls.value

    for key, value in missing_values_dict.items():
        missing_values_df[key] = value

    #append column to df with total missingness per column
    missing_values_df['Potential total missingness percentage in column'] = total_missing_in_column
    missing_values_df.rename(columns={"Percent of nulls in sample": "Percent missing"}, inplace=True)
    #show column names
    column_names = missing_values_df.index
    missing_values_df.insert(0, 'Columns', column_names)

    modified_list = [name.replace('"', '') for name in missing_values_df.columns]  # remove double quotes in values, like '"None"' --> 'None'
    missing_values_df.columns = modified_list
    # append the zeros, question marks and dashes list to the original dataframe, and sum
    missing_values_df.insert(len(missing_values_df.columns) - 1, 'Zeros', zeros)
    missing_values_df.insert(len(missing_values_df.columns) - 1, "?", question_marks)
    missing_values_df.insert(len(missing_values_df.columns) - 1, "-", dashes)
    missing_values_df['Potential total missingness percentage in column'] = [x + zeros_percentage_list[i] for i, x in enumerate(missing_values_df['Potential total missingness percentage in column'])]
    missing_values_df['Potential total missingness percentage in column'] = [x + question_marks_percentage_list[i] for i, x in
                                                                             enumerate(missing_values_df[
                                                                                           'Potential total missingness percentage in column'])]
    missing_values_df['Potential total missingness percentage in column'] = [x + dashes_percentage_list[i] for i, x in
                                                                             enumerate(missing_values_df[
                                                                                           'Potential total missingness percentage in column'])]
    missing_values_df['Potential total missingness percentage in column'] = missing_values_df['Potential total missingness percentage in column'].round(2)
    total_potential_missingness_sum = missing_values_df['Potential total missingness percentage in column'].sum()
    missing_values_df['Percent missing'] = round(missing_values_df['Percent missing']*100, 2)
    missing_values_df = dash_datatable_format_fix(missing_values_df)
    #if no missing values, display check passed
    if total_potential_missingness_sum == 0:
        missing_values_df = pd.DataFrame({"Check notification": ["Check passed: No missing values encountered"]})

    return missing_values_df #returns dataframe with missing values

def duplicates(df, dtypes):
    """"Check 1: Function that finds duplicate values"""
    columns_to_drop = []
    #if ID column exist or column with only unique values, do not include this is the check as this hinders finding duplicates
    for column in df.columns:
        if dtypes[column] == 'not-generalizable' and df[column].nunique() == len(df):
            columns_to_drop.append(column)
    df = df.drop(columns=columns_to_drop)
    #the following block is required as fillna gives an error when replacing missing values with ' ' if it is not in their original categories
    categorical_cols = []
    for col in df.columns:
        if str(df[col].dtype) == 'category':
            categorical_cols.append(col)
    for col in categorical_cols:
        df[col] = df[col].cat.add_categories(' ') #prevent error in pd.fillna() for column of type 'category'

    df = df.fillna(' ') #comparing NaN to NaN values returns false, so for this check replace with fictive missing value ' '
    #group df by all columns and check which occur more than once
    duplicate_df = df.groupby(df.columns.tolist()).apply(lambda x: list(x.index)).reset_index(name='indexes')
    duplicate_df = duplicate_df[duplicate_df['indexes'].apply(len) > 1]
    duplicate_df['amount of duplicates'] = duplicate_df['indexes'].apply(lambda x: len(x) - 1)
    duplicate_df['indexes'] = duplicate_df['indexes'].apply(lambda x: [i for i in x])

    #if the are no duplicates display this using the same convention as other checks
    if duplicate_df.empty:
        duplicate_df = pd.DataFrame({"Check notification": ["Check passed: No duplicate instances encountered"]})
        dq_issue_report_string = ' '
    if not 'Check notification' in list(duplicate_df.columns):
        total_duplicates = int(duplicate_df['amount of duplicates'].sum())
        dq_issue_report_string = f'{total_duplicates} duplicate instances were encountered, investigate and/or remove them.'
    #fix potential dash datatable issues
    final_df = dash_datatable_format_fix(duplicate_df)

    return final_df, dq_issue_report_string


def duplicate_column(df):
    """"Check 2: Duplicate columns Checks for columns that are exact duplicates """
    duplicate_cols = []
    result = []
    #loop over columns in dataframe and check whether they are duplicates and not already said to be duplicates
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2 and (col2, col1) not in duplicate_cols:
                if df[col1].equals(df[col2]):
                    duplicate_cols.append((col1, col2))
                    result.append({'Column': col1, 'Duplicate column': col2})

    if duplicate_cols:
        df_duplicate_columns = pd.DataFrame(result)
        df_duplicate_columns = dash_datatable_format_fix(df_duplicate_columns)
        duplicate_cols = len(pd.unique(df_duplicate_columns['Column']))
        return df_duplicate_columns, f'{duplicate_cols} duplicate columns were encountered, remove the redundant columns.'
    else:
        return pd.DataFrame({"Check notification": ["Check passed: No duplicate columns encountered"]}), ' '