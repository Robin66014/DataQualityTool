import dash
from dash import html, dcc
from dash import dash_table
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Data quality toolkit"

import pandas_dq
import pandas as pd
data = pd.read_csv('datasets\data.csv')
report = pandas_dq.dq_report(data, target=None, csv_engine="pandas", verbose=1)
reportJSON = report.to_json
reportDICT = report.to_dict()
table = pd.DataFrame.from_dict(reportDICT)

app.layout = html.Div(
    [
        # main app framework
        html.Div("Data quality toolkit", style={'fontSize':50, 'textAlign':'center'}),
        dash_table.DataTable(table.to_dict('records')),
        html.Hr(),

        # content of each page
        dash.page_container
    ]
)


if __name__ == "__main__":
    app.run(debug=True)