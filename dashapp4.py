import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H1("Button and Progress Bar Example"),
        dbc.Button(
            "Click me", id="example-button", className="me-2", n_clicks=0
        ),
        html.div(id='progress')
    ],
    className="mt-4",
)


@app.callback(
    Output("progress", "children"), [Input("example-button", "n_clicks")]
)
def on_button_click(n):
    if n is None:
        return "Not clicked."
    else:
        progress = progress_function()
        return progress


def progress_function():

    progress_bars = html.Div(
        [
            dbc.Progress(value=25, color="success", className="mb-3"),
            dbc.Progress(value=50, color="warning", className="mb-3"),
            dbc.Progress(value=75, color="danger", className="mb-3"),
            dbc.Progress(value=100, color="info"),
        ]
    )

    return progress_bars


if __name__ == '__main__':
    app.run_server(debug=True)