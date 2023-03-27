# TODO: Test with pytest-mpl in the future

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List


def plot_prediction(y_true: np.ndarray, y_pred: np.ndarray, image):
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
    sns.lineplot(data=false_values, x=false_values.index, y="Value", color="blue", label="Predicted")
    plt.title("True Values x Predicted Values")
    plt.savefig(image)


def plot_feature_importance(feature_importances: List[float], columns: List[str], image):
    importances = pd.DataFrame({
        "Importances": feature_importances,
        "Features": columns
    })

    plt.clf()
    sns.barplot(data=importances, x="Importances", y="Features")
    plt.savefig(image)
