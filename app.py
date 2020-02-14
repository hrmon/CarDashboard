from typing import List

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output

from load_data import get_dataframe

df = get_dataframe()

# remove fucking spam posts
df.drop(df[df.price > 10000000].index, inplace=True)
tdf = df[['state', 'price']].groupby('state').mean()

def model_brand_df():
    dd = df.groupby(['model', 'manufacturer'], as_index=False).count().sort_values('Unnamed: 0')
    dd.rename(columns={'Unnamed: 0': 'count'}, inplace=True)
    return dd[dd['count'] > 2000]

def count_by_manufacturer():
    dic = df.groupby('manufacturer').count()['Unnamed: 0'].to_dict()
    return {'x': list(dic.keys()), 'y': list(dic.values())}


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

wk_days = [
    'monday',
    'tuesday',
    'wednesday',
    'thursday',
    'friday',
    'saturday',
    'sunday',
]


def generate_dest_choro(dd_select):
    tdf = df[['state', 'price']].groupby('state').mean()

    return {"data": [go.Choropleth(
        locations=list(tdf.index.str.upper()),  # Spatial coordinates
        z=list(tdf['price'].astype(float)),  # Data to be color-coded
        locationmode='USA-states',  # set of locations match entries in `locations`
        colorscale='Reds',
        colorbar_title="USD",
    )], "layout": dict(
        title=dict(
            text='some title',
            font=dict(family="Open Sans, sans-serif", size=15, color="#515151"),
        ),
        margin=dict(l=20, r=20, b=20, pad=5),
        automargin=False,
        clickmode="event+select",
        geo=go.layout.Geo(
            scope="usa", projection=go.layout.geo.Projection(type="albers usa")
        ),
    )}


def generate_submission_time_hm(states: List[str], select=False):
    # Total flight count by Day of Week / Hour

    hm = []

    for i in range(24):
        hm_df = df[df.hour == i].groupby('day').count()['image_url']
        hm.append(hm_df)

    hm_df = pd.concat(hm, axis=1)

    y = list(wk_days)

    trace = dict(
        type="heatmap",
        z=hm_df.to_numpy(),
        x=list(f"{i}:00" for i in range(24)),
        y=y,
        colorscale=[[0, "#71cde4"], [1, "#ecae50"]],
        reversescale=True,
        showscale=True,
        xgap=2,
        ygap=2,
        colorbar=dict(
            len=0.7,
            ticks="",
            title="Submissions",
            titlefont=dict(family="Gravitas One", color="#515151"),
            thickness=15,
            tickcolor="#515151",
            tickfont=dict(family="Open Sans, sans serif", color="#515151"),
        ),
    )

    title = f"Submission time by days/hours State <b>{states}</b>"

    layout = dict(
        title=dict(
            text=title,
            font=dict(family="Open Sans, sans-serif", size=15, color="#515151"),
        ),
        font=dict(family="Open Sans, sans-serif", size=13),
        automargin=True,
    )

    return {"data": [trace], "layout": layout}


app.layout = html.Div(children=[
    html.H1(children='Hello Car Dashboard'),
    html.Div(children='DashDash: A web application framework for Python.'),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {**count_by_manufacturer(), 'type': 'bar', 'name': 'SF'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    ),
    html.Div(
        id="top-row",
        className="row",
        children=[
            html.Div(
                id="map_geo_outer",
                className="seven columns",
                # avg arrival/dep delay by destination state
                children=dcc.Graph(id="choropleth", figure={"data": [go.Choropleth(
                    locations=list(tdf.index.str.upper()),  # Spatial coordinates
                    z=list(tdf['price'].astype(float)),  # Data to be color-coded
                    locationmode='USA-states',  # set of locations match entries in `locations`
                    colorscale='Reds',
                    colorbar_title="USD",
                )], "layout": dict(
                    title=dict(
                        text='some title',
                        font=dict(family="Open Sans, sans-serif", size=15, color="#515151"),
                    ),
                    margin=dict(l=20, r=20, b=20, pad=5),
                    automargin=False,
                    clickmode="event+select",
                    geo=go.layout.Geo(
                        scope="usa", projection=go.layout.geo.Projection(type="albers usa")
                    ),
                )}),
            ),
        ],
    ),
    html.Div(
        id="bad-row",
        className="row",
        children=[

            html.Div(
                id="flights_by_day_hm_outer",
                className="five columns",
                children=dcc.Loading(children=dcc.Graph(id="submission_time_hm")),
            ),
        ],
    ),
    html.Div(
        id="bottom-row",
        className="row",
        children=html.Div(
            id="models_by_brand_sunburst",
            className="8 columns",
            children=[
                dcc.Loading(children=dcc.Graph(id="model_by_brand", figure=px.sunburst(
                    model_brand_df(),
                    path=['manufacturer', 'model'],
                    values='count',
                ))),
            ]
        )
    ),
])


@app.callback(
    Output("submission_time_hm", "figure"),
    [Input("choropleth", "clickData"), Input("choropleth", "figure")],
)
def update_hm(choro_click, choro_figure):
    if choro_click is not None:
        states = []
        for point in choro_click["points"]:
            states.append(point["location"])

        return generate_submission_time_hm(states, select=True)
    else:
        return generate_submission_time_hm([], select=False)


if __name__ == '__main__':
    app.run_server(debug=True)
