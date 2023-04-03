import dash
from dash import html, dcc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Data quality toolkit"
app.layout = html.Div(
    [
        # main app framework
        html.Div("Data quality toolkit", style={'fontSize':50, 'textAlign':'center'}),
        html.Hr(),

        # content of each page
        dash.page_container
    ]
)


if __name__ == "__main__":
    app.run(debug=True)