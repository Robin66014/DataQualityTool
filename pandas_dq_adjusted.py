import pandas as pd
import numpy as np
import copy
import os
def dq_report_adjusted(data, dtypes, mixed_data_types_df, target=None):
    """
    This is a data quality reporting tool that accepts any kind of file format as a filename or as a
    pandas dataframe as input and returns a report highlighting potential data quality issues in it.
    The function performs the following data quality checks. More will be added periodically.
     It detects missing values and suggests to impute them with mean, median,
      mode, or a constant value. It also identifies rare categories and suggests to group them
       into a single category or to drop them.
       The function finds infinite values and suggests to replace them with NaN or a
        large value. It detects mixed data types and suggests to convert them
        to a single type or split them into multiple columns.
         The function detects duplicate rows and columns, outliers in numeric columns,
          high cardinality features only in categorical columns, and
          highly correlated features.
    Finally, the function identifies if the problem is a classification problem or
     a regression problem and checks if there is class imbalanced or target leakage in the dataset.
    """

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('colheader_justify', 'center')

    df = copy.deepcopy(data)

    # Drop duplicate rows
    dup_rows = df.duplicated().sum()
    # if dup_rows > 0:
    #     print(f'    Alert: Dropping {dup_rows} duplicate rows can sometimes cause column data types to change to object. Double-check!')
    #     df = df.drop_duplicates()

    # Drop duplicate columns
    dup_cols = df.columns[df.columns.duplicated()]
    # if len(dup_cols) > 0:
    #     print(f'    Alert: Dropping {len(dup_cols)} duplicate cols')
    #     ### DO NOT MODIFY THIS LINE. TOOK A LONG TIME TO MAKE IT WORK!!!
    #     ###  THis is the only way that dropping duplicate columns works. This is not found anywhere!
    #     df = df.T[df.T.index.duplicated(keep='first')].T


    ### This is the column that lists our data quality issues
    new_col = 'DQ Issue'
    good_col = "The Good News"
    bad_col = "The Bad News"

    # Create an empty dataframe to store the data quality issues
    dq_df1 = pd.DataFrame(columns=[good_col, bad_col])
    dq_df1 = dq_df1.T
    dq_df1["first_comma"] = ""
    dq_df1[new_col] = ""

    # Create an empty dataframe to store the data quality issues
    data_types = pd.DataFrame(
        df.dtypes,
        columns=['Data Type']
    )

    missing_values = df.isnull().sum()
    missing_values_pct = ((df.isnull().sum()/df.shape[0])*100)
    missing_cols = missing_values[missing_values > 0].index.tolist()
    number_cols = df.select_dtypes(include=["integer", "float"]).columns.tolist() # Get numerical columns
    float_cols = df.select_dtypes(include=[ "float"]).columns.tolist() # Get float columns
    id_cols = []
    zero_var_cols = []

    missing_data = pd.DataFrame(
        missing_values_pct,
        columns=['Missing Values%']
    )
    unique_values = pd.DataFrame(
        columns=['Unique Values%']
    )
    for row in list(df.columns.values):
        if row in float_cols:
            unique_values.loc[row] = ["NA"]
        else:
            unique_values.loc[row] = [int(100*df[row].nunique()/df.shape[0])]
            if df[row].nunique() == df.shape[0]:
                id_cols.append(row)
            elif df[row].nunique() == 1:
                zero_var_cols.append(row)

    maximum_values = pd.DataFrame(
        columns=['Maximum Value']
    )
    minimum_values = pd.DataFrame(
        columns=['Minimum Value']
    )

    for col in list(df.columns.values):
        if col not in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                maximum_values.loc[col] = [df[col].max()]
        elif col in number_cols:
            maximum_values.loc[col] = [df[col].max()]

    for col in list(df.columns.values):
        if col not in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                minimum_values.loc[col] = [df[col].min()]
        elif col in number_cols:
            minimum_values.loc[col] = [df[col].min()]

    ### now generate the data quality starter dataframe
    dq_df2 = data_types.join(missing_data).join(unique_values).join(minimum_values).join(maximum_values)

    ### set up additional columns
    dq_df2["first_comma"] = ""
    dq_df2[new_col] = f""

    #### This is the first thing you need to do ###############
    if dup_rows > 0:
        new_string =  f"There are {dup_rows} duplicate rows in the dataset. De-Dup them using Fix_DQ."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
    else:
        new_string =  f"There are no duplicate rows in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '
    ### DO NOT CHANGE THE NEXT LINE. The logic for columns is different.
    if len(dup_cols) > 0:
        new_string =  f"There are {len(dup_cols)} duplicate columns in the dataset. De-Dup {dup_cols} using Fix_DQ."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
    else:
        new_string =  f"There are no duplicate columns in this datatset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect ID columns in dataset and recommend removing them
    if len(id_cols) > 0:
        new_string = f"There are ID columns in the dataset. Recommend removing them before modeling."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in id_cols:
            # Append a row to the dq_df1 with the column name and the issue only if the column has a missing value
            new_string = f"Possible ID colum: drop before modeling process."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string = f"There are no ID columns in the dataset. So no ID columns to remove before modeling."
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect ID columns in dataset and recommend removing them
    if len(zero_var_cols) > 0:
        new_string = f"There are zero-variance columns in the dataset. Recommend removing them before modeling."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in zero_var_cols:
            # Append a row to the dq_df1 with the column name and the issue only if the column has a missing value
            new_string = f"Zero-variance column: drop before modeling process."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string = f"There are no zero-variance columns in the dataset. So no zero-variance columns to remove before modeling."
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect missing values and suggests to impute them with mean, median, mode, or a constant value123
    #missing_values = df.isnull().sum()
    #missing_cols = missing_values[missing_values > 0].index.tolist()
    if len(missing_cols) > 0:
        for col in missing_cols:
            # Append a row to the dq_df1 with the column name and the issue only if the column has a missing value
            if missing_values[col] > 0:
                new_string = f"{missing_values[col]} missing values. Impute them with MICE, mean, median, mode, or a constant value."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string = f"There are no columns with missing values in the dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '


    # Identify rare categories and suggests to group them into a single category or drop them123
    rare_threshold = 0.01 # Define a 1% threshold for rare categories
    #cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist() # Get categorical columns
    cat_cols = []  # dataset.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in df.columns:
        if dtypes[col] == 'categorical' or dtypes[col] == 'boolean':
            cat_cols.append(col)

    rare_cat_cols = []
    if len(cat_cols) > 0:
        for col in cat_cols:
            value_counts = df[col].value_counts(normalize=True)
            rare_values = value_counts[value_counts < rare_threshold].index.tolist()
            if len(rare_values) > 0:
                rare_cat_cols.append(col)
                # Append a row to the dq_df2 with the column name and the issue
                if len(rare_values) <= 10:
                    new_string = f"{len(rare_values)} rare categories: {rare_values}. Group them into a single category or drop the categories."
                else:
                    new_string = f"{len(rare_values)} rare categories: Too many to list. Group them into a single category or drop the categories."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no categorical columns with rare categories (< {100*rare_threshold:.0f} percent) in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '


    # Find infinite values and suggests to replace them with NaN or a large value123
    inf_values = df.replace([np.inf, -np.inf], np.nan).isnull().sum() - missing_values
    inf_cols = inf_values[inf_values > 0].index.tolist()
    if len(inf_cols) > 0:
        new_string =  f"There are {len(inf_cols)} columns with infinite values in the dataset. Replace them with NaN or a finite value."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in inf_cols:
            if inf_values[col] > 0:
                new_string = f"{inf_values[col]} infinite values. Replace them with a finite value."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no columns with infinite values in this dataset "
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    first_row_numeric = pd.to_numeric(mixed_data_types_df.iloc[0], errors='coerce')  # convert strings to numeric
    mixed_types_deepchecks = (first_row_numeric > 0) & (first_row_numeric < 1)

    # Detect mixed data types and suggests to convert them to a single type or split them into multiple columns123
    mixed_cols = mixed_types_deepchecks[mixed_types_deepchecks == True].index.tolist() # Get the columns with more than one type
    if len(mixed_cols) > 0:
        new_string = f"There are {len(mixed_cols)} columns with mixed data types in the dataset. Convert them to a single type or split them into multiple columns."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in mixed_cols:
            if mixed_types_deepchecks[col] == True:

                new_string = f"Mixed dtypes: contains strings and numerical data"
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no columns with mixed (more than one) dataypes in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '


    # Detect outliers in numeric cols
    num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist() # Get numerical columns
    if len(num_cols) > 0:
        first_time = True
        outlier_cols = []
        for col in num_cols:
            q1 = df[col].quantile(0.25) # Get the first quartile
            q3 = df[col].quantile(0.75) # Get the third quartile
            iqr = q3 - q1 # Get the interquartile range
            lower_bound = q1 - 1.5 * iqr # Get the lower bound
            upper_bound = q3 + 1.5 * iqr # Get the upper bound
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col] # Get the outliers
            outliers_high = df[(df[col] > upper_bound)][col]
            outliers_lower = df[(df[col] < lower_bound)][col]
            if len(outliers) > 0:
                outlier_cols.append(col)
                if first_time:
                    new_string = f"There are {len(num_cols)} numerical columns, some with outliers. Remove them or use robust statistics."
                    dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
                    dq_df1.loc[bad_col,'first_comma'] = ', '
                    first_time =False
                ### check if there are outlier columns and print them ##
                new_string = f"has {len(outliers_high)} outliers greater than upper bound ({upper_bound:.2f}) and {len(outliers_lower)} lower than lower bound({lower_bound:.2f}). Cap them or remove them."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
        if len(outlier_cols) < 1:
            new_string =  f"There are no numeric columns with outliers in this dataset"
            dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
            dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect high cardinality features only in categorical columns
    cardinality_threshold = 100 # Define a threshold for high cardinality
    cardinality = df[cat_cols].nunique() # Get the number of unique values in each categorical column
    high_card_cols = cardinality[cardinality > cardinality_threshold].index.tolist() # Get the columns with high cardinality
    if len(high_card_cols) > 0:
        new_string = f"There are {len(high_card_cols)} columns with high cardinality (>{cardinality_threshold} categories) in the dataset. Reduce them using encoding techniques or feature selection methods."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in high_card_cols:
            new_string = f"high cardinality with {cardinality[col]} unique values: Use hash encoding or embedding to reduce dimension."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no high cardinality columns in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect highly correlated features
    correlation_threshold = 0.9 # Define a threshold for high correlation
    correlation_matrix = df[num_cols].corr().abs() # Get the absolute correlation matrix of numerical columns
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)) # Get the upper triangle of the matrix
    high_corr_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)] # Get the columns with high correlation
    if len(high_corr_cols) > 0:
        new_string = f"There are {len(high_corr_cols)} columns with >= {correlation_threshold} correlation in the dataset. Drop one of them or use dimensionality reduction techniques."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in high_corr_cols:
            new_string = f"has a high correlation with {upper_triangle[col][upper_triangle[col] > correlation_threshold].index.tolist()}. Consider dropping one of them."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no highly correlated columns in the dataset."
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # First see if this is a classification problem
    if target is not None:
        if isinstance(target, str):
            target_col = [target]
        else:
            target_col = copy.deepcopy(target) # Define the target column name

        cat_cols = df[target_col].select_dtypes(include=["object", "category"]).columns.tolist()

        ### Check if it is a categorical var, then it is classification problem ###
        model_type = 'Regression'
        if len(cat_cols) > 0:
            model_type =  "Classification"
        else:
            int_cols = df[target_col].select_dtypes(include=["integer"]).columns.tolist()
            copy_target_col = copy.deepcopy(target_col)
            for each_target_col in copy_target_col:
                if len(df[each_target_col].value_counts()) <= 30:
                    model_type =  "Classification"

        ### Then check for imbalanced classes in each target column
        if model_type == 'Classification':
            for each_target_col in target_col:
                y = df[each_target_col]
                # Get the value counts of each class
                value_counts = y.value_counts(normalize=True)
                # Get the minimum and maximum class frequencies
                min_freq = value_counts.min()
                max_freq = value_counts.max()
                # Define a threshold for imbalance
                imbalance_threshold = 0.1

                # Check if the class frequencies are imbalanced
                if min_freq < imbalance_threshold or max_freq > 1 - imbalance_threshold:
                    # Print a message to suggest resampling techniques or class weights
                    new_string =  f"Imbalanced classes in target variable ({each_target_col}). Use resampling or class weights to address."
                    dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
                    dq_df1.loc[bad_col,'first_comma'] = ', '

    # Detect target leakage in each feature
    if target is not None:
        target_col = copy.deepcopy(target) # Define the target column name
        if isinstance(target, str):
            preds = [x for x in list(df) if x not in [target_col]]
        else:
            preds = [x for x in list(df) if x not in target_col]
        leakage_threshold = 0.8 # Define a threshold for feature leakage
        leakage_matrix = df[preds].corrwith(df[target_col]).abs() # Get the absolute correlation matrix of each column with the target column
        leakage_cols = leakage_matrix[leakage_matrix > leakage_threshold].index.tolist() # Get the columns with feature leakage
        if len(leakage_cols) > 0:
            new_string = f"There are {len(leakage_cols)} columns with data leakage. Double check whether you should use this variable."
            dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
            dq_df1.loc[bad_col,'first_comma'] = ', '
            for col in leakage_cols:
                new_string = f"    {col} has a correlation >= {leakage_threshold} with {target_col}. Possible data leakage. Double check this variable."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
        else:
            new_string =  f'There are no target leakage columns in the dataset'
            dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
            dq_df1.loc[good_col,'first_comma'] = ', '
    else:
        new_string = f'There is no target given. Hence no target leakage columns detected in the dataset'
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    dq_df1.drop('first_comma', axis=1, inplace=True)
    dq_df2.drop('first_comma', axis=1, inplace=True)
    for col in list(df):
        if dq_df2.loc[col, new_col] == "":
            dq_df2.loc[col,new_col] += "No issue"


    from IPython.display import display
    all_rows = dq_df2.shape[0]
    ax = dq_df2.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
    display(ax);

        # Return the dq_df1 as a table
    return dq_df2
##################################################################################################

############################################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number =  '1.28'
#print(f"""{module_type} pandas_dq ({version_number}). Always upgrade to get latest features.""")
#################################################################################