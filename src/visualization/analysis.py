# TODO: Test with pytest-mpl in the future
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_prediction(y_true: np.ndarray, y_pred: np.ndarray):
    """ Plots predicted values

    Parameters:
    y_true: Labels of test
    y_pred: Predicted labels
    """
    y_true = np.squeeze(y_true)
    values = np.concatenate((y_true, y_pred), axis=0)
    labels = np.concatenate((["True"] * len(y_true), ["Predicted"] * len(y_pred)), axis=0)
    results = pd.DataFrame({
        "Value": values,
        "Label": labels
    })

    true_values = results[results["Label"] == "True"].reset_index()
    false_values = results[results["Label"] == "Predicted"].reset_index()

    plt.clf()
    sns.lineplot(data=true_values, x=true_values.index, y="Value", color="red", label="True")
    sns.lineplot(data=false_values, x=false_values.index, y="Value", color="blue", label="Predicted", alpha=0.4)
    plt.title("True Values x Predicted Values")
    plt.show()


def plot_feature_importance(feature_importances: List[float], columns: List[str]):
    """ Plots the feature importances

    Parameters:
    feature_importances: List of importance for each feature
    columns: List of columns in the dataframe
    """
    importances = pd.DataFrame({
        "Importances": feature_importances,
        "Features": columns
    })

    plt.clf()
    sns.barplot(data=importances, x="Importances", y="Features")
    plt.show()


def visualize_forecast_by_cluster(clusters_results: Dict[str, List]):
    """ Shows forescat results for each cluster

    Parameters:
    clusters_results: Dict containing cluster as key to a list. The first
    item of the list is the test data and the second is the predicted data
    """
    plt.clf()
    ncols = 2
    nrows = math.ceil(len(clusters_results)/2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))

    keys = sorted(clusters_results.keys())
    for index, axis in enumerate(axes.reshape(-1)):
        if index < len(clusters_results):
            current_key = keys[index]
            data = clusters_results[current_key]
            sns.lineplot(x=range(len(data[0])), y=data[0], marker='.', color='blue', label='Actual', ax=axis)
            sns.lineplot(x=range(len(data[1])), y=data[1], marker='^', color='green', label='Decision Tree', ax=axis)
            axis.set_xlabel("Date")
            axis.set_ylabel("Price")
            axis.set_title(f"Cluster {current_key}")
        else:  # If number of plots is odd, don't print the last one since it's expecting an even number of plots
            axis.axis('off')

    fig.tight_layout()
    plt.show()
