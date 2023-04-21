def generate_warning_report(df_list):
    # Create a list of the names of the dataframes with issues
    problem_dfs = [df_name for df_name in ["df_missing_values", "df_duplicates"] if df_name in globals()]

    # Count the number of problem dataframes
    num_problems = len(problem_dfs)

    # Generate a warning message based on the number of problems found
    if num_problems == 0:
        return "No issues found in dataset."
    elif num_problems == 1:
        return f"WARNING! 1 potential type of error has been found in your dataset ({problem_dfs[0]})."
    else:
        return f"WARNING! {num_problems} potential types of errors have been found in your dataset ({', '.join(problem_dfs)})."

    #TOTAL 5 warnings!
    #TODO: opzoeken in ydata-quality/profling hoe ze dit doen

    #CATEGORY 1: SEVERE DATA quality ISSUES
    #duplicate columns
    #missing values in target column
    #conflicting labels test
    #potentieel: wrong labels
    #mixed data types?
    #categorical value met meer dan 100 waardes oid? can create sparse data sets
    #wrong label?



    #CATEGORY 2: POTENTIALLY HARMFUL ISSUES
    #statistical normality tests
    #display whether features have not been normalized/standardized
    #missing values in input features
    #duplicates exact duplicates
    #outliers
    #collinearity (met de message Collinearity does not imply causality!)
    #special characters
    #string mismatch (potentially meant to be same value)
    #class imbalance
    #wrong label?
    #weinig data punten
    #sparsity / amount of features vs amount of instances (dimensionality)

    #alle dataframes met hun conclusie


