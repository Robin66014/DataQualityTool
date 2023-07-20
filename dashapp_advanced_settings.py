import dash
from dash import Dash
import dash_bootstrap_components as dbc
from dash import html
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]
)

app.layout =dbc.Accordion(
        [dbc.Card(
            dbc.AccordionItem(
                title="Advanced settings",
                children=[
                        dbc.Row([#missingess threshold
                        dbc.Col(dbc.Input(
                        id='advanced_settings_missing',  type='number',  value=5.0, min=0, max=100, ), width=2
                        ),
                        dbc.Col(html.H6("(%) Allowed missingness threshold [0,100]%"), align='center')]),

                        dbc.Row([#duplicate instances threshold
                        dbc.Col(dbc.Input(
                        id='advanced_settings_duplicates',  type='number',  value=1.0, min=0, max=100, ), width=2
                        ),
                        dbc.Col(html.H6("(%) Allowed duplicates threshold [0,100]%"), align='center')]),

                        dbc.Row([#outlier threshold
                        dbc.Col(dbc.Input(
                        id='advanced_settings_outliers',  type='number',  value=1.0, min=0, max=100, ), width=2
                        ),
                        dbc.Col(html.H6("(%) Allowed outliers threshold [0,100]%"), align='center')]),

                        dbc.Row([#correlation threshold
                        dbc.Col(dbc.Input(
                        id='advanced_settings_correlation',  type='number',  value=0.9, min=0, max=1, ), width=2
                        ),
                        dbc.Col(html.H6("(fraction) Allowed correlation threshold [0,1]"), align='center')]),

                        dbc.Row([#duplicate columns
                        dbc.Col(dbc.Switch(
                        id='advanced_settings_duplicate_columns',  value = False,), width=2
                        ),
                        dbc.Col(html.H6("Allow duplicate columns"), align='center')]),

                        dbc.Row([#single value
                        dbc.Col(dbc.Switch(
                            id='advanced_settings_single_value', value=False, ), width=2
                        ),
                        dbc.Col(html.H6("Allow columns with the same unique value in each row (redundancy)"), align='center')]),

                        dbc.Row([#mixed data types
                        dbc.Col(dbc.Switch(
                        id='advanced_settings_mixed_data_types',  value = False,), width=2
                        ),
                        dbc.Col(html.H6("Allow mixed data types in columns"), align='center')]),

                        dbc.Row([#special characters
                        dbc.Col(dbc.Switch(
                            id='advanced_settings_special_characters', value=False, ), width=2
                        ),
                        dbc.Col(html.H6("Allow data points consisting of only special characters (!@#$%)"), align='center')]),

                        dbc.Row([#string mismatch
                        dbc.Col(dbc.Switch(
                        id='advanced_settings_string_mismatch',  value = False,), width=2
                        ),
                        dbc.Col(html.H6("Allow string mismatches (e.g. 'Red', 'RED' & 'red)'"), align='center')]),

                        dbc.Row([#conflicting labels
                        dbc.Col(dbc.Switch(
                            id='advanced_settings_conflicting_labels', value=False, ), width=2
                        ),
                        dbc.Col(html.H6("Allow conflicting labels (same feature values, different label)"), align='center')]),

                        html.Br(), #blank line
                        dbc.Row([
                        dbc.Col(dbc.Switch(id='advanced_settings_10k', value = True, label='Sample 10.000 rows for smooth running time'))]),

                ], style={'backgroundColor': 'gray'}
            ), style={'backgroundColor': 'gray'})
        ],start_collapsed=True,
          # adjust as needed
)
if __name__ == "__main__":
    app.run_server(debug=True)