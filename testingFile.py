def box_plot(df, dtypes):

    numerical_columns = []
    for col in df.columns:
        # histogram for numerical values
        if dtypes[col] == 'floating' or dtypes[col] == 'integer':
            numerical_columns.append(col)
            # Add traces for each numerical column
    data = []
    for column in numerical_columns:
        data.append(go.Box(y=df[column], name=column))

    layout = go.Layout(
    title="Numerical Column Visualization for identifying column outliers",
    yaxis_title="Value"
                        )
    fig = go.Figure(data=data, layout=layout)


    return fig
import pandas as pd
import plotly.graph_objects as go

adult_df = pd.read_csv('datasets/adult1000.csv')
# def clean_dataset(df):
#     #same as from helper.py of openML
#     df = df.loc[:, df.isnull().mean() < 0.8]
#     out = df.fillna(df.mode().iloc[0])
#     return out
# df = clean_dataset(adult_df)
# dtypes_adult = {'age': 'integer', 'workclass': 'categorical', 'fnlwgt': 'integer', 'education': 'categorical', 'educational-num': 'integer', 'marital-status': 'categorical', 'occupation': 'categorical', 'relationship': 'categorical', 'race': 'categorical', 'gender': 'categorical', 'capital-gain': 'integer', 'capital-loss': 'integer', 'hours-per-week': 'integer', 'native-country': 'categorical', 'income': 'categorical'}
#
# fig = box_plot(df, dtypes_adult)
# fig.show()
