import deepchecks.tabular.checks
import pandas as pd
amount_of_samples = 1000000
amount_of_columns = 100000
df = pd.read_csv('datasets/creditcard.csv')

checkStringMismatch = deepchecks.tabular.checks.StringMismatch(n_top_columns=amount_of_columns,
                                                               n_samples=amount_of_samples)
resultStringMismatch = checkStringMismatch.run(df)

result = resultStringMismatch.display
print('result', result)

if result:
    df = pd.DataFrame(result[1])
    df = pd.DataFrame(df.to_records())  # flatten hierarchical index in columns
else:
    df = pd.DataFrame({"Message": ["No string mismatch or variants of the same string encountered"]})