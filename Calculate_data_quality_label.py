from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd
def calculate_dataset_nutrition_label(df, check_results):
    #categories to subdivide in: Critical issues, moderate issues, minor issues
    calculated_scores = {}
    rows_df, columns_df = df.shape
    max_missingness_percentage_allowed = 5#%
    max_outlier_percentage_allowed = 1
    feature_correlation_threshold = 0.9
    #TODO: thresholds gebruiken ipv numerieke waardes
    for check_res in check_results:
        if check_res == 'df_missing_values':
            if 'Check notification' in list(check_results[check_res].columns): #then the check is passed and 0 missing values were encountered
                calculated_scores['missing_values'] =  100
            else:
                missing_values_df = check_results['df_missing_values']
                column_count_check_passed = (missing_values_df['Percent missing (NA)'] < max_missingness_percentage_allowed).sum()
                calculated_scores['missing_values'] = (column_count_check_passed/columns_df)*100
        elif check_res == 'df_duplicates':
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['duplicate_instances'] =  100
            else:
                duplicates_df = check_results['df_duplicates']
                total_duplicates = duplicates_df['amount of duplicates'].sum()
                calculated_scores['duplicate_instances'] =  ((rows_df-total_duplicates)/rows_df)*100
        elif check_res == 'df_duplicate_columns':
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['duplicate_columns'] =  100
            else:
                duplicate_columns_df = check_results['df_duplicate_columns']
                duplicate_cols = len(pd.unique(duplicate_columns_df['Column']))
                calculated_scores['duplicate_columns'] =  ((columns_df-duplicate_cols)/columns_df)*100
        elif check_res == 'df_amount_of_diff_values':
            single_value_columns = (check_results['df_amount_of_diff_values'].iloc[0] == 1).sum()
            calculated_scores['amount_of_diff_values'] =  ((columns_df-single_value_columns)/columns_df)*100
        elif check_res == 'df_mixed_data_types':
            mixed_data_types_df = check_results['df_mixed_data_types']
            first_row_numeric = pd.to_numeric(mixed_data_types_df.iloc[0], errors='coerce') #convert strings to numeric
            mixed_columns = ((first_row_numeric > 0) & (first_row_numeric < 1)).sum() #check if columns contain mixed datatypes
            calculated_scores['mixed_data_types'] = ((columns_df-mixed_columns)/columns_df)*100
        elif check_res == 'df_special_character': #TODO: oranje maken?
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['special_characters'] =  100
            else:
                special_characters_df = check_results['df_special_character']
                calculated_scores['special_characters'] = ((columns_df-len(special_characters_df))/columns_df)*100
        elif check_res == 'df_string_mismatch': #TODO: oranje maken?
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['string_mismatch'] =  100
            else:
                calculated_scores['string_mismatch'] =  100 #TODO: wat hiermee doen?
        elif check_res == 'df_outliers': #TODO: oranje maken als score 1 is?
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['outliers'] =  100
            else:
                outliers_df = check_results['df_outliers']
                calculated_scores['outliers'] =  ((rows_df-len(outliers_df))/rows_df)*100
        elif check_res == 'df_feature_feature_correlation':
            feature_feature_correlation_df = check_results['df_feature_feature_correlation'].copy()
            feature_feature_correlation_df.drop(columns=['Column'], inplace=True)
            upper = feature_feature_correlation_df.where(np.triu(np.ones(feature_feature_correlation_df.shape), k=1).astype(bool))
            upper = upper.abs()
            high_corr_cols = [column for column in upper.columns if any(upper[column] > feature_correlation_threshold)] # Get the columns with high correlation
            correlation_dict = {}
            #find with which columns it is highly correlated
            for col in high_corr_cols:
                correlated_with_indices = upper[col][upper[col] > feature_correlation_threshold].index.tolist()
                correlated_with = [upper.columns[i] for i in correlated_with_indices]
                correlation_dict[col] = correlated_with

            calculated_scores['feature_correlations'] = ((columns_df-(len(high_corr_cols)))/columns_df)*100

        elif check_res == 'df_feature_label_correlation': #TODO: oranje maken?
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['feature_label_correlation'] =  100
            else:
                feature_label_correlation_df = check_results['df_feature_label_correlation']
                high_corr_cols = [column for column in feature_label_correlation_df.columns if any(feature_label_correlation_df[column] > feature_correlation_threshold)]
                calculated_scores['feature_label_correlation'] = ((columns_df-(len(high_corr_cols)))/columns_df)*100

        elif check_res == 'df_class_imbalance': #TODO: war hiermee doen
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['class_imbalance'] =  100
            else:
                class_imbalance_df = check_results['df_class_imbalance']
                max_value = class_imbalance_df.max().max()
                min_value = class_imbalance_df.min().min()
                ratio_min_max = min_value/max_value
                calculated_scores['class_imbalance'] =  ratio_min_max * 100

        elif check_res == 'df_conflicting_labels':
            if 'Check notification' in list(check_results[check_res].columns):
                calculated_scores['conflicting_labels'] =  1
            else:
                conflicting_labels_df = check_results['df_conflicting_labels']
                conflicting_labels_df['Conflicting'] = conflicting_labels_df['Instances'].str.count(',') + 1
                total_conflicting_labels = conflicting_labels_df['Conflicting'].sum()
                print(total_conflicting_labels)
                print(rows_df)
                calculated_scores['conflicting_labels'] =  (1 - (total_conflicting_labels/rows_df))*100
    print(calculated_scores)
    return calculated_scores
