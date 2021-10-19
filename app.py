from config import *
import ast
import base64
import json
import os
import sys

import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from itertools import combinations
from yellowbrick.classifier import  ClassificationReport, ROCAUC, PrecisionRecallCurve, ConfusionMatrix

from utils.helpers import create_heatmap
from utils.load_data import load_data


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import resample

if len(sys.argv) == 2:
    run_id = sys.argv[1]

report_df = pd.read_csv(OUTPUT_DATA_FILEPATH + 'report_df.csv').set_index('classifier')

panel_style = {
    'borderBottom': '1px solid #aadeee',
    'borderTop': '1px solid #aadeee',
    'padding': '10px',
    'borderRadius': '30px',
    'overflow': 'hidden',
    'backgroundColor': '#aadeee',
    'color': '#ffd369',
    'borderColor': '#aadeee',
    "margin-left": "40px",
    "margin-right": "40px",
    "margin-bottom": "40px",
    "margin-top": "40px"
    # 'fontWeight': 'bold'
}
sub_panel_style = {
    'borderBottom': '1px solid #aadeee',
    'borderTop': '1px solid #aadeee',
    'padding': '10px',
    'borderRadius': '30px',
    'overflow': 'hidden',
    'backgroundColor': '#aadeee',
    'color': '#ffd369',
    'borderColor': '#aadeee',
    "margin-left": "20px",
    "margin-right": "20px",
    "margin-bottom": "40px",
    "margin-top": "40px"
    # 'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #222831',
    'borderBottom': '1px solid #222831',
    'backgroundColor': '#ffd369',
    'color': '#222831',
    'padding': '10px',
    'borderRadius': '400px',
    'overflow': 'hidden',
    "margin-left": "4px"
}

# create Dash app
app = dash.Dash(external_stylesheets=[dbc.themes.YETI])

# helper function to render images
def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())

header_ =     html.Div(
                    className="app-header",
                    children=[
                        html.H1('CLASSIFIER DASHBOARD', className="app-header--title", style={'color': '#ce6453', 'font': 'San Francisco font', 'fontWeight': 'bold'}),
                    ])

app.layout = html.Div([
            header_,
            dcc.Graph(id='heatmap-graph', figure=create_heatmap(report_df), style=panel_style),
            html.Img(id='ROCAUC-image', src='children', height=300, style=sub_panel_style),
            html.Img(id='PrecisionRecallCurve-image', src='children', height=300, style=sub_panel_style),
            html.Img(id='ClassificationReport-image', src='children', height=300, style=sub_panel_style),
            html.Img(id='ConfusionMatrix-image', src='children', height=300, style=sub_panel_style),
            html.Div([
                html.Pre(id='hover-data', style={'paddingTop':35})
                ], style={'width':'30%'})
            ], style={'backgroundColor':'#2c3e50'})

@app.callback(
    [Output('ROCAUC-image', 'src'), Output('PrecisionRecallCurve-image', 'src'),
    Output('ClassificationReport-image', 'src'), Output('ConfusionMatrix-image', 'src')],
    [Input('heatmap-graph', 'hoverData')])
def callback_image(hoverData):
    # set path to current working directory
    path = os.getcwd() + '/'

    # interprets hover data json as dictionary
    hover_dict = ast.literal_eval(json.dumps(hoverData, indent=2))
    model_on_hover = hover_dict['points'][0]['y']

    # create list of yellowbrick visualizers with naming conventions matching how they are stored in img diretory
    visualizations = VISUALIZERS

    # create dictionary with each value as an element of visualizations list and value as associated base65 image
    output_list = []
    for visualization in visualizations:
        visualization_path = IMG_OUTPUT_FILEPATH  + model_on_hover + '/' + visualization + '.png'
        visualization_image = encode_image(path+visualization_path)
        output_list.append(visualization_image)
    return output_list

if __name__ == '__main__':
    app.run_server()
