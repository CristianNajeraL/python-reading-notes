"""
Dash app
"""

from typing import List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px


GITHUB_LINK = 'https://github.com/PacktPublishing/' \
              'Interactive-Dashboards-and-Data-Apps-with-Plotly-and-Dash'
WORLD_BANK_LINK = 'https://datacatalog.worldbank.org/dataset/poverty-and-equity-database'
GINI = 'GINI index (World Bank estimate)'
ID_VARS = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']

poverty_data = pd.read_csv("data/PovStatsData.csv")
country = pd.read_csv('data/PovStatsCountry.csv', na_values='', keep_default_na=False)

population_2010 = dict(
    zip(
        poverty_data[
            (poverty_data["Indicator Name"] == "Population, total")].loc[:, "Country Name"].values,
        poverty_data[(poverty_data["Indicator Name"] == "Population, total")].loc[:, "2010"].values
    )
)

data_melt = poverty_data.melt(id_vars=ID_VARS, var_name='year').dropna(subset=['value'])
data_melt['year'] = data_melt['year'].astype(int)
data_pivot = data_melt.pivot(
    index=['Country Name', 'Country Code', 'year'],
    columns='Indicator Name',
    values='value'
).reset_index()
poverty = pd.merge(
    data_pivot,
    country,
    left_on='Country Code',
    right_on='Country Code',
    how='left'
)

gini_df = poverty[poverty[GINI].notna()]
REGIONS = [
    'East Asia & Pacific', 'Europe & Central Asia', 'Fragile and conflict affected situations',
    'High income', 'IDA countries classified as fragile situations', 'IDA total',
    'Latin America & Caribbean', 'Low & middle income', 'Low income',
    'Lower middle income', 'Middle East & North Africa', 'Middle income',
    'South Asia', 'Sub-Saharan Africa', 'Upper middle income', 'World'
]
population_df = poverty_data[
    ~poverty_data['Country Name'].isin(REGIONS) &
    (poverty_data['Indicator Name'] == 'Population, total')
]

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
            value="World",
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
        dcc.Dropdown(
            id='year_dropdown',
            value='2010',
            options=[
                {
                    'label': year,
                    'value': str(year)
                } for year in range(1974, 2019)
            ]
        ),
        dcc.Graph(id='population_chart'),
        html.Br(),
        html.H2(
            'Gini Index - World Bank Data',
            style={
                'textAlign': 'center'
            }
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            'By Year',
                            style={
                                'textAlign': 'center'
                            }
                        ),
                        dcc.Dropdown(
                            id='gini_year_dropdown',
                            options=[
                                {
                                    'label': year,
                                    'value': year
                                } for year in gini_df['year'].drop_duplicates().sort_values()
                            ]
                        ),
                        dcc.Graph(
                            id='gini_year_barchart'
                        )
                    ],
                    style={
                        "width": "50%"
                    }
                ),
                dbc.Col(
                    [
                        html.H3(
                            'By Country',
                            style={
                                'textAlign': 'center'
                            }
                        ),
                        dcc.Dropdown(
                            id='gini_country_dropdown',
                            options=[
                                {
                                    'label': country,
                                    'value': country
                                } for country in gini_df['Country Name'].unique()
                            ]
                        ),
                        dcc.Graph(
                            id='gini_country_barchart'
                        )
                    ],
                    style={
                        "width": "50%"
                    }
                )
            ],
        ),
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
def display_country_report(country_to_show: str) -> List:
    """
    Generates population report by country
    :param country_to_show: Selected country in the dropdown
    :return: Country report
    """
    return [
        html.H3(country_to_show),
        f"The population of {country_to_show} in 2010 was {population_2010[country_to_show]:,.0f}."
    ]


@app.callback(
    Output("population_chart", "figure"),
    Input("year_dropdown", "value")
)
def plot_countries_by_population(year):
    """
    Generates figure with countries population
    :param year: Year to look up
    :return: Population figure
    """
    year_df = population_df[['Country Name', year]].sort_values(year, ascending=False)[:20]
    fig = go.Figure()
    fig.add_bar(x=year_df['Country Name'], y=year_df[year])
    fig.layout.title = f'Top twenty countries by population - {year}'
    fig.update_layout(title_x=0.5)
    return fig


@app.callback(
    Output('gini_year_barchart', 'figure'),
    Input('gini_year_dropdown', 'value')
)
def plot_gini_year_barchart(year):
    """
    Generates figure with GINI index by year
    :param year: Year to be shown
    :return: Gini figure
    """
    if not year:
        raise PreventUpdate
    df = gini_df[gini_df['year'].eq(year)].sort_values(GINI).dropna(subset=[GINI])
    n_countries = len(df['Country Name'])
    fig = px.bar(
        df,
        x=GINI,
        y='Country Name',
        log_x=True,
        orientation='h',
        height=200 + (n_countries*20),
        title=GINI + ' ' + str(year)
    )
    fig.update_layout(title_x=0.5)
    return fig


@app.callback(
    Output('gini_country_barchart', 'figure'),
    Input('gini_country_dropdown', 'value')
)
def plot_gini_country_barchart(country_to_show):
    """
    Generates figure with GINI index by country
    :param country_to_show: Country from the list
    :return: Gini figure
    """
    if not country_to_show:
        raise PreventUpdate
    df = gini_df[gini_df['Country Name'] == country_to_show].dropna(subset=[GINI])
    n_years = [str(i) for i in list(df["year"].unique())]
    fig = px.bar(
        df,
        x=GINI,
        y=n_years,
        labels={
            "y": "Year"
        },
        log_x=True,
        title=' - '.join([GINI, country_to_show]),
        orientation="h",
        height=200 + (len(n_years) * 20),
    )
    fig.update_layout(title_x=0.5)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
