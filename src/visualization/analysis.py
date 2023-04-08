# TODO: Test with pytest-mpl in the future
import math
from typing import Dict, List

import seaborn as sns
import matplotlib.pyplot as plt


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
