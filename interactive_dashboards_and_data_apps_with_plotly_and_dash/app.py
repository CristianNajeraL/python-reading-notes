"""
Dash app
"""

import dash
from dash import html
import dash_bootstrap_components as dbc


GITHUB_LINK = 'https://github.com/PacktPublishing/' \
              'Interactive-Dashboards-and-Data-Apps-with-Plotly-and-Dash'
WORLD_BANK_LINK = 'https://datacatalog.worldbank.org/dataset/poverty-and-equity-database'

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP
    ]
)

app.layout = html.Div(
    [
        html.H1(
            "Poverty And Equity Database",
            style={
                "color": "blue",
                "fontSize": "40px"
            }
        ),
        html.H2("The World Bank"),
        dbc.Tabs(
            [
                dbc.Tab(
                    [
                        html.Ul(
                            [
                                html.Li("Number of Economies: 170"),
                                html.Li("Temporal Coverage: 1974 - 2019"),
                                html.Li("Update Frequency: Annual"),
                                html.Li("Last Updated: March 18, 2020"),
                                html.Li(
                                    [
                                        'Source: ',
                                        html.A(
                                            'World Bank',
                                            href=WORLD_BANK_LINK,
                                            target="_blank",
                                            rel="noopener noreferrer"
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    label="Key Facts"
                ),
                dbc.Tab(
                    [
                        html.Ul(
                            [
                                html.Br(),
                                html.Li(
                                    'Book title: Interactive Dashboards and '
                                    'Data Apps with Plotly and Dash'
                                ),
                                html.Li(
                                    [
                                        html.A(
                                            'GitHub repo',
                                            href=GITHUB_LINK,
                                            target="_blank",
                                            rel="noopener noreferrer"
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    label="Project Info"
                )
            ]
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
