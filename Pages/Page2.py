import dash
from dash import html, dcc, dash_table, callback
import plotly.express as px
import settings
import dash_mantine_components as dmc
import duplicates_and_missing
import pandas as pd


dash.register_page(__name__)

#data = {'col1': [pd.NA, pd.NaT], 'col2': ['test', pd.NaT], 'col3': ['1', 'cat']}
#df = pd.DataFrame(data=data) #TODO: glitch met pd.NaT en pd.NA, alles geprobeerd.
df = pd.read_csv('datasets\iris.csv')
#df = settings.uploaded_file_df
df_missing_values = duplicates_and_missing.missing_values(df)
#TODO: verdergaan bij data uit file upload button gebruiken op andere pagina via chatGPT manier
# , als dat niet lukt omvormen tot automatisch naar beneden scrollen / hideable divs

# df_missing_values = df_missing_values.reset_index()
#df_missing_values = df_missing_values.columns.map(str)
# df_missing_values = df_missing_values.to_string()
layout = html.Div(
    [
        html.Button('Run function', id='button'),
        html.Div(id='output'),
        dcc.Link(html.Button('Previous page'), href=dash.page_registry['pages.Page1']['path']),
        dmc.Accordion(
            children=[
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Duplicates & missing values ({})".format(36)),
                        dmc.AccordionPanel([dash_table.DataTable(df_missing_values.to_dict('records')),
                            "Colors, fonts, shadows and many other parts are customizable to fit your design needs"]
                        ),

                    ],
                    value="duplicatesandmissing",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Type integrity ({})".format(36)),
                        dmc.AccordionPanel(
                            "Configure temp appearance and behavior with vast amount of settings or overwrite any part of "
                            "component styles "
                        ),
                    ],
                    value="typeintegrity",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Outliers & correlations ({})".format(36)),
                        dmc.AccordionPanel(
                            "Configure temp appearance and behavior with vast amount of settings or overwrite any part of "
                            "component styles "
                        ),
                    ],
                    value="outliersandcorrelations",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Label purity ({})".format(36)),
                        dmc.AccordionPanel(
                            "Configure temp appearance and behavior with vast amount of settings or overwrite any part of "
                            "component styles "
                        ),
                    ],
                    value="labelpurity",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Bias & fairness ({})".format(36)),
                        dmc.AccordionPanel(
                            "Configure temp appearance and behavior with vast amount of settings or overwrite any part of "
                            "component styles "
                        ),
                    ],
                    value="biasandfairness",
                ),
            ],
        )
    ]
)

# define the callback for the button click event
@callback(
    dash.dependencies.Output('output', 'children'),
    dash.dependencies.Input('button', 'n_clicks')
)
def run_function(n_clicks):
    #global first_time
    if settings.first_time:
        #TODO: Accordion met functies per categorie + pandas warnings

        # run the function here
        # df = pd.read_csv('datasets\data.csv')
        # df_duplicates = duplicates_and_missing.duplicates(df)
        print('Function has been run.')
        # set to false so the functions don't run again unless a new file is uploaded.
        settings.first_time = False
        return 'Function has been run.'
    else:
        return 'Function has already been run.'


#TODO: warnings from ydata-profiling