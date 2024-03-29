import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import train_test_split
from pandas_dq_adjusted import dq_report_adjusted
def pandas_dq_report(dataset, dtypes, mixed_data_types_df, special_characters_df, df_string_mismatch, df_feature_feature_correlation, target):
    """"Function to generate an extended version of pandas_dq report by incorporating all of the checks taken
     into account in calculating the data readiness level"""

    if target != 'None':
        #label encode target if categorical
        le = LabelEncoder()
        le.fit(dataset[target])
        #transform target column using the fitted encoder
        dataset[target] = le.transform(dataset[target])
        #report = pandas_dq.dq_report(dataset, target=target, csv_engine="pandas", verbose=1)
        report = dq_report_adjusted(dataset, dtypes, mixed_data_types_df, special_characters_df, df_string_mismatch, df_feature_feature_correlation, target=target)
    else:
        report = dq_report_adjusted(dataset, dtypes, mixed_data_types_df, special_characters_df, df_string_mismatch, df_feature_feature_correlation, target=None)
    #Convert to dict
    reportDICT = report.to_dict()
    #fix string issue (dtype) in pandas_dq conversion to a dictionary
    reportDICT = {k: {k2: str(v2).replace("dtype(", "dtype") for k2, v2 in v.items()} for k, v in reportDICT.items()}

    #convert to df and append list of column names to beginning of df
    reportDF = pd.DataFrame(reportDICT)
    reportDF.insert(0, 'Column', list(dataset.columns))
    return reportDF



def encode_categorical_columns(dataset, target, dtypes, mixed_dtypes_dict):
    """"Function that one-hot-encodes categorical columns and label encodes the target column. It returns the encoded dataset
    and the mapping of the original labels to the encoded labels"""
    #Find all categorical columns and mixed dtype column
    categorical_cols = []
    mixed_dtype_columns = []
    for col in dataset.columns:
        if dtypes[col] == 'categorical' or dtypes[col] == 'boolean':
            categorical_cols.append(col)
        if mixed_dtypes_dict[col] == True:
            mixed_dtype_columns.append(col)

    if mixed_dtype_columns: #drop mixed dtype columns to prevent encoder bugs
        dataset = dataset.drop(columns=mixed_dtype_columns)
        categorical_cols = [col for col in categorical_cols if col not in mixed_dtype_columns]

    mapping = None
    target_is_categorical = False
    if target != 'None':
        # remove target as we want to label encode this (for classification problems)
        if target in categorical_cols:
            target_is_categorical = True
            categorical_cols.remove(target)

            # label encode target
            le = LabelEncoder()
            encoded_target = le.fit_transform(dataset[target])
            encoded_labels = le.transform(dataset[target])
            mapping = {label: value for label, value in zip(encoded_labels, dataset[target])}
            # replace target column with label encoded values
            dataset.drop(columns=[target], inplace=True)
            dataset[target] = encoded_target
    if not categorical_cols:  # then no features are categorical, and we're done
        return dataset, mapping

    #if there are categorical columns, we want to one-hot-encode them
    #encode categorical columns, if more than 100 columns then categories will be combined
    encoder = OneHotEncoder(handle_unknown='ignore', max_categories=100)
    encoded_columns = encoder.fit_transform(dataset[categorical_cols])
    new_columns = pd.DataFrame(encoded_columns.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

    #add new columns to df and drop old ones
    dataset_encoded = pd.concat([dataset, new_columns], axis=1)
    dataset_encoded = dataset_encoded.drop(columns=categorical_cols)

    #reposition target column to the end of the dataframe
    if target != 'None' and target_is_categorical:
        dataset_encoded.drop(columns=[target], inplace=True)
        dataset_encoded[target] = encoded_target

    #XGBClassifier doesn't accept: [, ] or <, so loop over the columns and change the names if they contain such values
    new_col_names = {col: col.replace('<', '(smaller than)').replace('[', '(').replace(']', ')') for col in dataset_encoded.columns}
    dataset_encoded = dataset_encoded.rename(columns=new_col_names)

    return dataset_encoded, mapping

def label_encode_dataframe(df, dtypes):
    """"Label encodes dataframe that contain a target feature"""
    label_encoded_dict = {}
    #place holder message for label encoder mapping
    mapping_df = pd.DataFrame({"Message": ["No label encoding took place"]}) #as a dataframe to prevent future erros with label_mapping/mapping_df.to_dict()
    for column in df.columns:
        if dtypes[column] == 'categorical' or dtypes[column] == 'boolean' and not is_numeric_dtype(df[column]):
            unique_values = df[column].unique()
            label_encoded_dict[column] = {}
            for i, value in enumerate(unique_values):
                label_encoded_dict[column][value] = i
            df[column] = df[column].map(label_encoded_dict[column])
            # Create a DataFrame from the nested dictionary
            mapping_df = pd.DataFrame.from_dict([(key, tuple(val.items())) for key, val in label_encoded_dict.items()])
            mapping_df.columns = ['Column name', 'Encoding mapping']
            #modifications to the dataframe to makeit Dash.datatable-proof as it doesnt accept lists
            mapping_df['Encoding mapping'] = mapping_df['Encoding mapping'].apply(lambda x: str(x)[1:-1])

    return df, mapping_df

def pcp_plot(encoded_df, target, outlier_prob_scores):
    """"Creates a Parallel Coordinates Plot color coded on outlier probability score to inspect and verify outliers"""
    #if there are probability scores, color code on them
    if not outlier_prob_scores.empty:
        pcp_df = pd.concat([encoded_df, outlier_prob_scores], axis=1)
        fig = px.parallel_coordinates(pcp_df, color='Outlier Probability Score')
    else: #no outlier scores obtained, then just color code on target feature
        if target != 'None':
            fig = px.parallel_coordinates(encoded_df, color=target)
        else:
            fig = px.parallel_coordinates(encoded_df)

    return fig

def plot_feature_importance(df, target, dtypes):
    """"check 18: plots Random forest feature importances in a horizontal barchart, based on target encoded feature values"""
    te = TargetEncoder() #use target encoder as numerical values are required and one-hot-encoding gives results for each individual category
    #split target from data
    x = df.drop(columns=[target])
    y = df[target]
    #for classification problems, label encode the target and use RandomForestClassifier
    if dtypes[target] == 'boolean' or dtypes[target] == 'categorical':

        le = LabelEncoder()
        y = le.fit_transform(y)
        x = te.fit_transform(x, y)
        rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        rf.fit(x, y)
    else: #for regression problems, use RandomForestClassifier
        x = te.fit_transform(x, y)
        rf = RandomForestRegressor(n_estimators=10, n_jobs=-1)
        rf.fit(x, y)

    #obtain feature importances
    feature_importances = rf.feature_importances_

    #create df from them
    df_feature_importances = pd.DataFrame({
        'feature': x.columns,
        'importance': feature_importances
    })

    #horizontal barchart
    fig = px.bar(df_feature_importances.sort_values('importance', ascending=False),
                 x='importance', y='feature', orientation='h',
                 title='Feature Importance')
    return fig


def plot_dataset_distributions(data, dtypes):
    """"check 17: plots the distributions per column in the dataset"""

    figs = []

    # each datatypre requires different plotting
    for col in data.columns:
        # histogram for numerical values
        if dtypes[col] == 'floating' or dtypes[col] == 'integer' or dtypes[col] == 'numeric':
            fig = px.histogram(data, x=col, title=f'{col} distribution')
            figs.append(fig)

        #bar chart for categoricals & booleans
        elif dtypes[col] == 'boolean':
            fig = px.bar(data, x=col, color=col, title=f'{col} distribution')
            figs.append(fig)
        elif dtypes[col] == 'categorical':
            fig = px.bar(data, x=col, title=f'{col} distribution')
            figs.append(fig)

        #for sentences, plot the word frequency
        elif dtypes[col] == 'sentence':
            word_counts = data[col].str.split(expand=True).stack().value_counts()
            fig = px.bar(x=word_counts.index, y=word_counts.values, title=f'{col} word cloud')
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
            figs.append(fig)

    return figs


def baseline_model_performance(dataset, target, dtypes):
    """"check 19: Check the performance of the dataset for three simple sklearn models"""

    #check whether this is a regression/classification problem or has no task
    if target != 'None':
        if dtypes[target] == 'categorical' or dtypes[target] == 'boolean':
            problem_type = 'classification'
        else:
            problem_type = 'regression'
    else:
        return 'Dataset does not contain a target variable, so model training is not applicable'

    #train models for regression or classification
    models = []
    if problem_type == 'regression':
        models.append(('Linear Regression', LinearRegression(n_jobs=-1)))
        models.append(('Random Forest', RandomForestRegressor(n_jobs=-1)))
        models.append(('SVM', SVR()))
    elif problem_type == 'classification':
        models.append(('Logistic Regression', LogisticRegression(n_jobs=-1)))
        models.append(('Random Forest', RandomForestClassifier(n_jobs=-1)))
        models.append(('SVM', SVC()))

    #prevent errors with object columns
    object_columns = dataset.select_dtypes(include=['object']).columns
    dataset = dataset.drop(columns=object_columns)

    #get X and y and split into train and test for fair model evaluation
    X = dataset.drop(columns=[target])

    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #train models and get accuracies
    MSEs = []
    accuracies = []
    model_names = []
    for name, model in models:
        if problem_type == 'regression': #we use MSE for model evaluation
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            model_names.append(name)
            MSEs.append(mse)
        else: #we use accuracy for model evaluation
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = round(accuracy_score(y_test, y_pred), 2)
            model_names.append(name)
            accuracies.append(acc)

    #plot accuracies / MSE
    if problem_type == 'regression':
        fig = px.bar(x=MSEs, y=model_names, text=MSEs, labels={'x': 'Mean Squared Error', 'y': 'Model'},
                     color=model_names)
    else:
        fig = px.bar(x=accuracies, y=model_names, text=accuracies, labels={'x': 'Accuracy', 'y': 'Model'},
                     color=model_names)

    # Put values on the right side of the horizontal bars
    fig.update_traces(textposition='outside')

    return fig

def clean_dataset(df):
    """function for basic missing value removal for visualizations"""
    #same method as in helper.py of openML, remove columns with high missiness percentage
    df = df.loc[:, df.isnull().mean() < 0.8]
    #fill remaining column missing values with mode
    out = df.fillna(df.mode().iloc[0])
    return out


def dash_datatable_format_fix(df):
    """"Function to prevent dash from crashing due to formatting errors"""
    # fix dash datatables issues, as it does not accept dicts
    remove_brackets = lambda x: str(x).replace('[', '').replace(']', '').replace('{', '').replace('}', '')

    # apply the lambda function to the dataframe
    df = df.applymap(remove_brackets)

    return df


def upsample_minority_classes(df, target):
    """Copies instances where target class count is less than 5 and duplicates them until 5 is reached
     (which is necessary for label error prediction, 5-fold cross validation splits)."""
    classes = df[target].unique()
    dfs = []
    rare_classes = 0
    error_message = ''
    #check for each class if it contains more than 5 instances
    for class_value in classes:
        class_df = df[df[target] == class_value] #filter dataframe to contain only the class_value class


        if len(class_df) < 5:
            #upsample if this is true
            rare_classes += 1
            class_df = class_df.sample(n=5, replace=True)

        dfs.append(class_df)
    if rare_classes > 0:
        error_message = 'WARNING: {} rare classes have been detected. These instances have been upsampled to the minimum of 5 instances per class' \
                        ' (just for this check).'.format(rare_classes)
    #concatenate all the dataframes
    upsampled_df = pd.concat(dfs, axis=0)

    #shuffle the rows to mix the classes
    upsampled_df = upsampled_df.sample(frac=1).reset_index(drop=True)

    return upsampled_df, error_message