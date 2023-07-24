
import numpy as np
import pandas as pd
def calculate_dataset_nutrition_label(df, check_results, settings_dict):
    """"Function that calculates the data readiness level based on the check results"""

    calculated_scores = {}
    rows_df, columns_df = df.shape

    #colour scheme for dbc.Progress bars
    check_failed = 'danger'
    check_warning = 'warning'
    check_passed = 'success'

    #scoring mechanism DQ_label, categories to subdivide in: Critical issues, moderate issues, minor issues
    penalty_points = 0
    critical = 3
    moderate = 2
    minor = 1

    #for each check result, determine whether the check is passed, if not: penalize
    for check_res in check_results:
        if check_res == 'df_missing_values': #CRITICAL ISSUE
            calculated_scores['missing_values'] =  round((100-check_results['df_missing_values']),2)
            if calculated_scores['missing_values'] < (100-settings_dict['advanced_settings_missing']):
                calculated_scores['missing_values_color'] = check_failed
                penalty_points += critical
            else:
                calculated_scores['missing_values_color'] = check_passed

        elif check_res == 'df_duplicates': #CRITICAL ISSUE
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['duplicate_instances'] =  100
                total_duplicates = 0
            else:
                duplicates_df = check_results['df_duplicates']
                duplicates_df['amount of duplicates'] = pd.to_numeric(duplicates_df['amount of duplicates'])
                total_duplicates = int(duplicates_df['amount of duplicates'].sum())
                calculated_scores['duplicate_instances'] =  round(((rows_df-total_duplicates)/rows_df)*100,2)

            if (total_duplicates/rows_df)*100 > settings_dict['advanced_settings_duplicates']:
                calculated_scores['duplicate_instances_color'] = check_failed
                penalty_points += critical
            else:
                calculated_scores['duplicate_instances_color'] = check_passed

        elif check_res == 'df_duplicate_columns': #CRITICAL ISSUE
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['duplicate_columns'] =  100
                duplicate_cols = 0
            else:
                duplicate_columns_df = check_results['df_duplicate_columns']
                duplicate_cols = len(pd.unique(duplicate_columns_df['Column']))
                calculated_scores['duplicate_columns'] =  round(((columns_df-duplicate_cols)/columns_df)*100, 2)

            if duplicate_cols > 0 and settings_dict['advanced_settings_duplicate_columns'] == False: #there are duplicate columns and user does not allow
                calculated_scores['duplicate_columns_color'] = check_failed
                penalty_points += critical
            else:
                calculated_scores['duplicate_columns_color'] = check_passed

        elif check_res == 'df_amount_of_diff_values': #MINOR ISSUE
            single_value_columns = int((check_results['df_amount_of_diff_values'].iloc[0] == '1').sum()) #values in table are strings instead of integers
            calculated_scores['amount_of_diff_values'] = round(((columns_df - single_value_columns) / columns_df) * 100,
                                                               2)

            if single_value_columns > 0 and settings_dict['advanced_settings_single_value'] == False: #there is a redundant column and user does not allow
                calculated_scores['amount_of_diff_values_color'] = check_failed
                penalty_points += minor
            else:
                calculated_scores['amount_of_diff_values_color'] = check_passed


        elif check_res == 'df_mixed_data_types': #MODERATE ISSUE
            mixed_data_types_df = check_results['df_mixed_data_types']
            first_row_numeric = pd.to_numeric(mixed_data_types_df.iloc[0], errors='coerce') #convert strings to numeric
            mixed_columns = int(((first_row_numeric > 0) & (first_row_numeric < 1)).sum()) #check if columns contain mixed datatypes
            calculated_scores['mixed_data_types'] = round(((columns_df-mixed_columns)/columns_df)*100,2)

            if mixed_columns > 0 and settings_dict['advanced_settings_mixed_data_types'] == False: #there are columns with mixed dtypes and user does not
                calculated_scores['mixed_data_types_color'] = check_failed
                penalty_points += moderate
            else:
                calculated_scores['mixed_data_types_color'] = check_passed

        elif check_res == 'df_special_characters': #MINOR ISSUE
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['special_characters'] =  100
                special_characters_df = []
            else:
                special_characters_df = check_results['df_special_characters']
                calculated_scores['special_characters'] = round(((columns_df-len(special_characters_df))/columns_df)*100,2)
            if len(special_characters_df) > 0 and settings_dict['advanced_settings_mixed_data_types'] == False: #there are special characters and user does not allow
                calculated_scores['special_characters_color'] = check_failed
                penalty_points += minor
            else:
                calculated_scores['special_characters_color'] = check_passed #no special characters or they are allowed by user

        elif check_res == 'df_string_mismatch': #MINOR ISSUE
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['string_mismatch'] =  100
                calculated_scores['string_mismatch_color'] = check_passed
            else:
                columns_with_stringmismatch = check_results[check_res]['Column Name'].nunique()
                calculated_scores['string_mismatch'] =  round(((columns_df-columns_with_stringmismatch)/columns_df)*100, 2)
                if settings_dict['advanced_settings_string_mismatch'] == False:
                    calculated_scores['string_mismatch_color'] = check_failed
                    penalty_points += moderate
                elif settings_dict['advanced_settings_string_mismatch'] == True: #then string mismatched are allowed by user
                    calculated_scores['string_mismatch_color'] = check_passed

        elif check_res == 'df_outliers': #MINOR ISSUE
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['outliers'] =  100
                outliers_df = []
            else:
                outliers_df = check_results['df_outliers']
                calculated_scores['outliers'] =  round(((rows_df-len(outliers_df))/rows_df)*100,2)

            if (len(outliers_df)/rows_df)*100 > settings_dict['advanced_settings_outliers']:
                calculated_scores['outliers_color'] = check_failed
                penalty_points += minor
            else:
                calculated_scores['outliers_color'] = check_passed

        elif check_res == 'df_feature_feature_correlation': #MINOR ISSUE
            feature_feature_correlation_df = check_results['df_feature_feature_correlation'].copy()

            if 'Column' in feature_feature_correlation_df.columns:
                feature_feature_correlation_df.drop(columns=['Column'], inplace=True)

            upper = feature_feature_correlation_df.where(np.triu(np.ones(feature_feature_correlation_df.shape), k=1).astype(bool))
            upper = upper.apply(pd.to_numeric, errors='coerce')
            high_corr_cols = [column for column in upper.columns if (any(upper[column] > settings_dict['advanced_settings_correlation']) or any(upper[column] < -1*settings_dict['advanced_settings_correlation']))] # Get the columns with high correlation
            correlation_dict = {}
            #find with which columns it is highly correlated
            for col in high_corr_cols:
                correlated_with = upper[col][upper[col] > settings_dict['advanced_settings_correlation']].index.tolist()
                correlation_dict[col] = correlated_with

            calculated_scores['feature_correlations'] = round(((columns_df-(len(high_corr_cols)))/columns_df)*100,2)

            if len(high_corr_cols) > 0:
                calculated_scores['feature_correlations_color'] = check_failed
                penalty_points += minor
            else:
                calculated_scores['feature_correlations_color'] = check_passed

        elif check_res == 'df_feature_label_correlation': #NOT TAKEN INTO DQ LABEL
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['feature_label_correlation'] =  100
                calculated_scores['feature_label_correlation_color'] = check_passed
            else:
                feature_label_correlation_df = check_results['df_feature_label_correlation']
                high_corr_cols = [column for column in feature_label_correlation_df.columns if float(feature_label_correlation_df[column]) > settings_dict['advanced_settings_correlation']]#[column for column in feature_label_correlation_df.columns if any(float(feature_label_correlation_df[column]) > feature_correlation_threshold)]
                calculated_scores['feature_label_correlation'] = round(((columns_df-(len(high_corr_cols)))/columns_df)*100,2)
                calculated_scores['feature_label_correlation_color'] = check_warning

        elif check_res == 'df_class_imbalance': #NOT TAKEN INTO DQ LABEL
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['class_imbalance'] =  100
                calculated_scores['class_imbalance_color'] = check_passed
            else:
                class_imbalance_df = check_results['df_class_imbalance']
                max_value = float(class_imbalance_df.max().max())
                min_value = float(class_imbalance_df.min().min())
                ratio_min_max = min_value/max_value
                calculated_scores['class_imbalance'] =  round((ratio_min_max * 100))

            if calculated_scores['class_imbalance'] <= 10:
                calculated_scores['class_imbalance_color'] = check_warning
            else:
                calculated_scores['class_imbalance_color'] = check_passed

        elif check_res == 'df_conflicting_labels': #MODERATE ISSUE
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['conflicting_labels'] =  100
                total_conflicting_labels = 0
            else:
                conflicting_labels_df = check_results['df_conflicting_labels']
                conflicting_labels_df['Conflicting'] = conflicting_labels_df['Instances'].str.count(',') + 1
                total_conflicting_labels = int(conflicting_labels_df['Conflicting'].sum())
                calculated_scores['conflicting_labels'] =  round((1 - (total_conflicting_labels/rows_df))*100,2)

            if total_conflicting_labels > 0 and settings_dict['advanced_settings_conflicting_labels'] == False: #there are conflicting label and user does not allow them
                calculated_scores['conflicting_labels_color'] = check_failed
                penalty_points += moderate
            else: #there are no conflictin labels, or user allows them
                calculated_scores['conflicting_labels_color'] = check_passed

    #sum up penalty points to determine Data readiness label
    if penalty_points < 3:
        DQ_label = 'A'
    elif penalty_points >= 3 and penalty_points <6:
        DQ_label = 'B'
    elif penalty_points >= 6 and penalty_points <9:
        DQ_label = 'C'
    elif penalty_points >= 9 and penalty_points <12:
        DQ_label = 'D'
    elif penalty_points >= 12:
        DQ_label = 'E'

    return calculated_scores, DQ_label

