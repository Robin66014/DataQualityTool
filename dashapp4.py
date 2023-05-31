import os
import pandas as pd
from flask_caching import Cache
import uuid
import dash
import io
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
# Initialize the Dash app
app = dash.Dash(__name__)
current_directory = os.path.dirname(os.path.realpath('dashapp4.py'))
cache_dir = os.path.join(current_directory, 'cached_files')
# Configure the cache
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': cache_dir  # Directory where cache files will be stored
})

# Helper function to generate a unique filename based on session ID and pathname
def generate_filename(session_id, pathname):
    return f"{session_id}_{pathname.replace('/', '_')}.pkl"

# Example callback that fetches the data based on the session ID and unique pathname
@app.callback(
    [Output('output-div', 'children'),Output('stored-filepath', 'data')],
    [Input('url', 'pathname'),Input('upload-data', 'contents')]
)
@cache.memoize()
def fetch_data(pathname, contents):
    session_id = str(uuid.uuid4())
    filename = generate_filename(session_id, pathname)
    filepath = os.path.join(cache_dir, filename) #TODO: deze extern opslaan
    print('pathname', pathname)
    print('session_id', session_id)
    print('filename', filename)
    print('filepath', filepath)
    print('@@@@@@contents', contents)
    if os.path.exists(filepath):
        # If the file already exists, load the DataFrame from the cache
        df = pd.read_pickle(filepath)
        return f"Loaded DataFrame from cache: {df.to_string()}", filepath

    elif contents:
        #If the file doesn't exist but there is uploaded content, fetch the data and store it in the cache
        #Example: Read the uploaded CSV file contents and create a DataFrame
        content_type, content_string = contents.split(',')
        decoded_content = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))

        # Save the DataFrame to the cache directory
        df.to_pickle(filepath)
        return f"Stored DataFrame in cache: {df.to_string()}", filepath

    else:
        # No cache file or uploaded content available
        return "No data available", filepath

# Layout of the Dash app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        )
    ]),
    html.Div(id='output-div'),
    html.Button('Show Cached Data', id='show-button'),
    html.Div(id='table-div'),
    dcc.Store(id='stored-filepath', storage_type='memory')
])


# Example callback that displays the cached data when the button is clicked
@app.callback(
    Output('table-div', 'children'),
    [Input('show-button', 'n_clicks'),
     State('stored-filepath', 'data')]
)
def display_cached_data(n_clicks, filepath):
    if n_clicks:
        if os.path.exists(filepath):
            # If the file already exists, load the DataFrame from the cache
            df = pd.read_pickle(filepath)
            return f"Loaded DataFrame from cache123: {df.to_string()}"


if __name__ == '__main__':
    app.run_server(debug=True)