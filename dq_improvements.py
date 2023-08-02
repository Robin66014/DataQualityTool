import pandas as pd
import re
from pandas.api.types import is_numeric_dtype
import numpy as np

def impute_missing_values(df, dtypes, imputation_methods):
    df_previous = df.copy()
    imputation_methods = imputation_methods[0] #get the dict out of the list
    for column in list(dtypes.keys()):
        if column not in df.columns or df[column].empty: #then it has been deleted before or the whole column is empty and we cant impute
            continue
        if dtypes[column] == 'categorical' or dtypes[column] == 'boolean':
            if imputation_methods[column] == 'most frequent':
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif imputation_methods[column] == 'do not impute':
                print('No imputation performed for column {}'.format(column))
            elif imputation_methods[column] in ['mean', 'median']:
                if is_numeric_dtype(df[column]): #can happen that sortinghat identifies a numerical column as catgorical due to the
                    # low nr of distinct values, we should then notify the user about his imputation choice
                    if imputation_methods[column] == 'mean':
                        df[column].fillna(df[column].mean(), inplace=True)
                        print('Column {} imputed using mean strategy, but "most frequent" is recommended as it is a most probably'
                              ' a categorical column with numeric values.'.format(column))
                    elif imputation_methods[column] == 'median':
                        df[column].fillna(df[column].median(), inplace=True)
                        print('Column {} imputed using median strategy, but "most frequent" is recommended as it is a most probably'
                              ' a categorical column with numeric values.'.format(column))
                else:
                    print("Invalid imputation strategy for column {}: use 'most frequent' instead. "
                      "The missing values in this column will not be imputed.".format(column))

        elif dtypes[column] == 'floating' or dtypes[column] == 'integer' or dtypes[column] == 'numeric':
            if imputation_methods[column] == 'most frequent':
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif imputation_methods[column] == 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
            elif imputation_methods[column] == 'median':
                df[column].fillna(df[column].median(), inplace=True)
            elif imputation_methods[column] == 'do not impute':
                print('No imputation performed for column {}'.format(column))
        else: #other data types than numeric or categorical
            try:
                if imputation_methods[column] == 'most frequent':
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif imputation_methods[column] == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif imputation_methods[column] == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif imputation_methods[column] == 'do not impute':
                    print('No imputation performed for column {}'.format(column))
            except Exception as e:
                print(e)
                print('Error occured with column {}. Column will not be imputed'.format(column))


    changed_data = []
    diff_mask = (df != df_previous)
    diff_df = df[diff_mask]
    diff_series = diff_df.stack(dropna=True)
    # Iterate over the multi-index Series
    for idx, new_val in diff_series.iteritems():
        row_idx, col_name = idx
        row_idx_new = df.loc[row_idx, 'Original Index']
        previous = df_previous.loc[row_idx, col_name]
        change = {
            "coordinates": (row_idx_new, col_name),
            "new": new_val}
        # "previous": previous}
        changed_data.append(change)

    return changed_data




def fix_string_mismatch(df, df_string_mismatch, value_to_replace):

    df_previous = df
    result = {}

    #obtain base forms and the variants of the base form
    for key in df_string_mismatch['Base form'].keys():
        base_form = df_string_mismatch['Base form'][key]
        value = df_string_mismatch['Value'][key]

        if base_form not in result:
            result[base_form] = []
        result[base_form].append(value)

    for key, values in result.items():
        if key == value_to_replace:
            for value in values:
                df = df.replace(value, key)

    changed_data = []
    diff_mask = (df != df_previous)
    diff_df = df[diff_mask]
    diff_series = diff_df.stack(dropna=True)
    #find changed
    for idx, new_val in diff_series.iteritems():
        row_idx, col_name = idx
        row_idx_new = df.loc[row_idx, 'Original Index']
        previous = df_previous.loc[row_idx, col_name]
        change = {
            "coordinates": (row_idx_new, col_name),
            "new": new_val}
            #"previous": previous}
        changed_data.append(change)

    return changed_data


def fix_special_characters(df, df_special_characters, replacement_value):
    df_previous = df
    #obtain unique string values with only special charcters
    #get all special characters
    special_char_list = list(df_special_characters['Most Common Special-Only Samples'].values())

    #split based on comma and obtain unique samples
    samples_split = [item.strip().strip("'") for sublist in special_char_list for item in sublist.split(',')]
    unique_values = list(set(samples_split))

    # unique_values = set(value.strip("' ") for s in df_special_characters['Most Common Special-Only Samples'] for value in s.split(','))
    print('@@@unique values', unique_values)
    print('@@@replacement_value', replacement_value)
    df = df.replace(unique_values, replacement_value)

    changed_data = []
    diff_mask = (df != df_previous)
    diff_df = df[diff_mask]
    diff_series = diff_df.stack(dropna=True)
    #find changes
    for idx, new_val in diff_series.iteritems():
        row_idx, col_name = idx
        row_idx_new = df.loc[row_idx, 'Original Index']
        previous = df_previous.loc[row_idx, col_name]
        change = {
            "coordinates": (row_idx_new, col_name),
            "new": new_val}
        # "previous": previous}
        changed_data.append(change)

    return changed_data



def obtain_indices_with_issues(df, check_results, settings_dict):
    dq_issue_indices = {}
    dq_issue_indices['redundant_columns'] = []
    for check_res in check_results:
        if check_res == 'df_missing_values':
            missing_value_mask = df.isna()
            missing_value_indices = missing_value_mask[missing_value_mask.any(axis=1)].index.tolist()
            if len(missing_value_indices) > 0:
                dq_issue_indices['missing_values'] = missing_value_indices
            else:
                dq_issue_indices['missing_values'] = []
        elif check_res == 'df_duplicates':
            if 'Check notification' in list(check_results[check_res].columns):
                dq_issue_indices['duplicate_instances'] = []
            else:
                df_duplicates = check_results['df_duplicates']
                #split string and get all but the first index (as we do not want to delete that one)
                df_duplicates['indexes'] = df_duplicates['indexes'].str.split(',')
                # dq_issue_indices['duplicate_instances'] = df_duplicates['indexes'].apply(lambda x: x[1:])
                # #convert to int
                # dq_issue_indices['duplicate_instances'] = dq_issue_indices['duplicate_instances'].apply(lambda x: [int(i) for i in x])
                dq_issue_indices['duplicate_instances'] = df_duplicates['indexes'].apply(lambda x: x[1:]).tolist()
                dq_issue_indices['duplicate_instances'] = [int(item) for sublist in
                                                           dq_issue_indices['duplicate_instances'] for item in sublist]
        elif check_res == 'df_duplicate_columns':
            if not 'Check notification' in list(check_results[check_res].columns):
                if len(check_results[check_res]['Duplicate column'].values) > 0:
                    dq_issue_indices['redundant_columns'].append(check_results[check_res]['Duplicate column'].values)

        elif check_res == 'df_amount_of_diff_values':
            df_single_value = check_results['df_amount_of_diff_values']
            columns = df_single_value.columns[df_single_value.iloc[0] == 1].tolist()
            if len(columns) > 0:
                dq_issue_indices['redundant_columns'].append(columns)
        elif check_res == 'df_mixed_data_types':
            mixed_data_types_df = check_results['df_mixed_data_types']
            first_row_numeric = pd.to_numeric(mixed_data_types_df.iloc[0], errors='coerce') #convert strings to numeric
            mask = (first_row_numeric > 0) & (first_row_numeric < 1)  #check if columns contain mixed datatypes (less than 100% of one data type)
            mixed_columns = mixed_data_types_df.columns[mask]
            if not mixed_columns.empty:
                dq_issue_indices['redundant_columns'].append(mixed_columns.tolist())
        elif check_res == 'df_special_characters':
            if not 'Check notification' in list(check_results[check_res].columns):
                indices = []
                columns_with_specials = check_results[check_res]['Column'].values
                special_samples = set(value.strip("' ") for s in check_results[check_res]['Most Common Special-Only Samples'] for value in
                    s.split(','))
                for col in columns_with_specials:
                    mask = df[col].isin(special_samples)
                    col_indices = df.index[mask].tolist()
                    indices.append(col_indices)
                flattened_list = [item for sublist in indices for item in sublist]
                dq_issue_indices['special_characters'] = flattened_list

            else:
                dq_issue_indices['special_characters'] = []

        elif check_res == 'df_string_mismatch': #MINOR ISSUE
            if not 'Check notification' in list(check_results[check_res].columns):
                df_string_mismatch = check_results[check_res]
                values = df_string_mismatch['Value'].tolist()
                mask = df.isin(values)
                mask_series = mask.any(axis=1)
                indices = df[mask_series].index.tolist()
                dq_issue_indices['string_mismatch'] = indices
            else:
                dq_issue_indices['string_mismatch'] = []

        elif check_res == 'df_outliers': #MINOR ISSUE
            if 'Check notification' in list(check_results[check_res].columns):
                dq_issue_indices['outlier_instances'] = []
            else:
                indices = check_results[check_res]['Row number'].values
                indices_string = list(indices)
                dq_issue_indices['outlier_instances'] = [int(i) for i in indices_string]

        elif check_res == 'df_feature_feature_correlation': #MINOR ISSUE
            feature_feature_correlation_df = check_results['df_feature_feature_correlation'].copy()

            if 'Column' in feature_feature_correlation_df.columns:
                feature_feature_correlation_df.drop(columns=['Column'], inplace=True)

            upper = feature_feature_correlation_df.where(
                np.triu(np.ones(feature_feature_correlation_df.shape), k=1).astype(bool))
            upper = upper.apply(pd.to_numeric, errors='coerce')

            high_corr_cols = set()
            #check which columns have higher correlation than specified by user
            for col in upper.columns:
                for row in upper.index:
                    if abs(upper.loc[row, col]) > settings_dict['advanced_settings_correlation']:
                        high_corr_cols.add(row)
                        high_corr_cols.add(col)

            high_corr_cols = list(high_corr_cols)

        elif check_res == 'df_conflicting_labels': #MODERATE ISSUE
            if 'Check notification' in list(check_results[check_res].columns):
                dq_issue_indices['conflicting_labels'] = []
            else:
                indices = check_results[check_res]['Instances'].str.split(',').apply(lambda x: list(map(int, x))).sum()
                dq_issue_indices['conflicting_labels'] = indices

    dq_issue_indices['redundant_columns'] = list(set([item for sublist in dq_issue_indices['redundant_columns'] for item in list(sublist)]))

    return dq_issue_indices