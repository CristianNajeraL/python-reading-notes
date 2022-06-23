"""
Dash app
"""

from typing import List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output

GITHUB_LINK = 'https://github.com/PacktPublishing/' \
              'Interactive-Dashboards-and-Data-Apps-with-Plotly-and-Dash'
WORLD_BANK_LINK = 'https://datacatalog.worldbank.org/dataset/poverty-and-equity-database'
poverty_data = pd.read_csv("data/PovStatsData.csv")
population_2010 = dict(
    zip(
        poverty_data[
            (poverty_data["Indicator Name"] == "Population, total")].loc[:, "Country Name"].values,
        poverty_data[(poverty_data["Indicator Name"] == "Population, total")].loc[:, "2010"].values
    )
)

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
        dcc.Dropdown(
            id="country",
            options=[
                {
                    "label": country,
                    "value": country
                } for country in poverty_data["Country Name"].unique()
            ]
        ),
        html.Div(
            id="report"
        ),
        html.Br(),
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


@app.callback(
    Output("report", "children"),
    Input("country", "value")
)
def display_country_report(country: str) -> List:
    """
    Generates population report by country
    :param country: Selected country in the dropdown
    :return: Country report
    """
    if country is None:
        return [
            html.H3("World"),
            f"The world population in 2010 was {population_2010['World']:,.0f}."
        ]
    return [
        html.H3(country),
        f"The population of {country} in 2010 was {population_2010[country]:,.0f}."
    ]


if __name__ == "__main__":
    app.run_server(debug=True)
