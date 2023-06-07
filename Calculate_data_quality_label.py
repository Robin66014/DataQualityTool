#TODO: the grande finale
from pandas.api.types import is_numeric_dtype
def calculate_dataset_nutrition_label(check_results):
    #categories to subdivide in: Critical issues, moderate issues, minor issues

    if list(check_results['missing_values_check'].columns) == ['Message']: #then the check is passed and 0 missing values were encountered
        missing_values_score = 1
    else:
        missing_values_df = check_results['missing_values_check']






    return label
