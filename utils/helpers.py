import ast
import pandas as pd
import numpy as np
import yellowbrick
import os
from pandas.io.json import json_normalize
import plotly.graph_objects as go
from sklearn.metrics import classification_report as classificationreport
from utils.visualizers import Visualizer
from utils import colors



def create_img(X_train, X_test, y_train, y_test, labels, model, visualizer,  IMG_OUTPUT_FILEPATH):
    """
    Create and save yellowbrick visualizer image for model specificed

    Args:
        X_train: numpy ndarray of model features training data values
        X_test: numpy ndarray of model features test data values
        y_train: numpy ndarray of model target variable training data values
        y_test: numpy ndarray of model target variable test data values
        labels: list of class labels for binary classification
        visualizer: yellowbrick classification visualizer
        IMG_OUTPUT_FILEPATH (str): filepath name for output image

    """
    viz = Visualizer(X_train, X_test, y_train, y_test, labels, model, visualizer)
    viz.evaluate()
    outpath_ = IMG_OUTPUT_FILEPATH + str(model).split('(')[0] + '/' + visualizer + '.png'
    viz.visualizer.show(outpath=outpath_, clear_figure=True)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluates model by providing individual category and summary metrics of model performance
    Args:
        model: sklearn estimator for classification
        X_train: numpy ndarray of model features training data values
        X_test: numpy ndarray of model features test data values
        y_train: numpy ndarray of model target variable training data values
        y_test: numpy ndarray of model target variable test data values

    Returns:
        report: classification report with evaluation metrics (f1, precision, recall, support)

    """
    model = model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classificationreport(y_test, y_pred, target_names= ["0", "1"], output_dict=True)

    return report



def customize_dict_keys(x, col, keys):
    """
    Revise key values to personalize to its associated column i.e. from 'precision' to 'precision_0'

    Args:
        x: pandas series
        col (str): column name of pandas series
        keys: list containing string values of classification metrics

    Returns:
        new_dict: dictionary containing new key names from string concatenation

    """
    new_keys = [key + '_' + col for key in keys]
    new_dict = dict(zip(new_keys, list(x.values())))
    return new_dict

def normalize_to_flat(classifier, df, col):
    """
    Normalize json data into flat table

    Args:
        classifier (str): index value of report_df dataframe named as classifier model
        df: pandas dataframe
        col (str): column name of pandas series

    Returns:
        new_df: pandas dataframe of flat table

    """
    name = str(classifier) + '_df'
    new_df = json_normalize(df.loc[classifier][col])
    new_df['classifier'] = [classifier]
    return new_df


def revise_report_df(report_df):
    """

    Transform report_df dataframe to format useable for displaying heatmap

    Args:
        report_df: pandas dataframe of sklearn classification_report output row-wise for by each model

    Returns:
        report_df: pandas dataframe formatted for use by create_heatmap function

    """
    # quick check to see whether report_df column structure is as expected
    if report_df.columns.tolist() != ['0', '1', 'accuracy', 'macro avg', 'weighted avg']:
        print("Warning: Column names aren't as expected. Verify report_df output_dict is correct.")

    report_df.columns = ['0', '1', 'accuracy', 'Macro Avg', 'Micro Avg' ]

    dict_columns = ['0', '1', 'Macro Avg', 'Micro Avg']
    keys = ['precision', 'recall', 'f1-score', 'support']

    for col in dict_columns:
        # revise key values to personalize to its associated column i.e. from 'precision' to 'precision_0'
        report_df[col] = report_df[col].apply(lambda x: customize_dict_keys(x, col, keys))

        # iterate row wise through dataframe to normalize dictionary values into flat tables
        new_dict = {str(classifier) + '_df': normalize_to_flat(classifier, report_df, col) for classifier in report_df.index.values.tolist()}

        # concat all classifier flat tables into one dataframe
        dict_df = pd.concat(list(new_dict.values())).reset_index().drop(columns=['index'], axis=1)

        # merge on existing report_df dataframe index and on dict_df 'classifier' column value
        report_df = report_df.merge(dict_df, how='left', left_index=True, left_on=None, right_on='classifier').set_index('classifier')

    # select only created columns
    report_df = report_df.iloc[:,5:]
    # sort columns and filter out 'support' related columns
    report_df = report_df[sorted([col for col in report_df.columns if 'support' not in col])]

    return report_df


def create_heatmap(df):

    """
    Create Plotly Heatmap graph object

    Args:
        df: transformed report_df pandas dataframe from the process_data.py file

    Returns:
        fig: ploty Figure graph object

    """

    fig = go.Figure(data=go.Heatmap(
                       z=df.values.tolist(),
                       x=df.columns,
                       #y=[classifier for classifier in df.index.values.tolist()],
                        y = df.index.values.tolist(),
                       hoverongaps = False,
                        xgap = 3,
                        ygap = 3,
                        colorscale=colors.get_colorscale()
                        ),
                       )
    fig.update_layout(
        paper_bgcolor='#aadeee',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#2c3e50'
    )
    return fig
