import pandas as pd
from scipy.stats import shapiro, anderson, kstest


def test_normality(dataset, column_types):
    """
    Performs normality tests on all numeric columns of a dataset and returns the test results.

    Args:
        dataset (pandas DataFrame): The input dataset to test.
        column_types (list): A list of column data types for the dataset.

    Returns:
        pandas DataFrame: A dataframe with the test results for all numeric columns.
    """
    # Create an empty dataframe to store the test results
    test_results = pd.DataFrame(columns=['column', 'shapiro_wilk_stat', 'shapiro_wilk_pvalue',
                                         'anderson_stat', 'anderson_crit_vals', 'anderson_sig_levels',
                                         'kolmogorov_smirnov_stat', 'kolmogorov_smirnov_pvalue'])

    # Iterate through each column in the dataset
    for col in dataset.columns:
        if column_types[col] == 'numeric':
            data = dataset[col]

            # Perform the Shapiro-Wilk test
            shapiro_result = shapiro(data)

            # Perform the Anderson-Darling test
            anderson_result = anderson(data)

            # Perform the Kolmogorov-Smirnov test
            ks_result = kstest(data, 'norm')

            # Add the results to the test_results dataframe
            test_results = test_results.append({'column': col,
                                                'shapiro_wilk_stat': shapiro_result.statistic,
                                                'shapiro_wilk_pvalue': shapiro_result.pvalue,
                                                'anderson_stat': anderson_result.statistic,
                                                'anderson_crit_vals': anderson_result.critical_values,
                                                'anderson_sig_levels': anderson_result.significance_level,
                                                'kolmogorov_smirnov_stat': ks_result.statistic,
                                                'kolmogorov_smirnov_pvalue': ks_result.pvalue},
                                               ignore_index=True)
    return test_results


