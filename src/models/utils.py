from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# TODO: Add tests for these method
def evaluate_cv_search(search: RandomizedSearchCV):
    """ Evaluates the trained XGBoost model uing randomized search

    Parameters:
    search: The randomized search object
    """
    print("Best parameters:", search.best_params_)
    print("Lowest RMSE: ", np.sqrt(-search.best_score_))
    print("Feature importance:", search.best_estimator_.feature_importances_)


def get_clusters(dataframe: pd.DataFrame, n_clusters = 10, ignore_cols: List[str] = []) -> Union[pd.DataFrame, KMeans]:
    """ Assign each house in the dataset a cluster

    Parameters:
    dataframe: Dataframe to be labeled
    n_clusters: Number of clusters to split
    ignore_cols: List of columns to ignore while creating the clusters

    Returns:
    dataframe: Returns the labeled dataframe
    kmeans: Returns the trained KMeans algortihm to be used in future predictions
    """
    data = dataframe.drop(ignore_cols, axis="columns")

    kmeans = KMeans(n_clusters)
    labels = kmeans.fit_predict(data)

    dataframe["cluster"] = labels
    return dataframe, kmeans


def get_dataframes_by_cluster(dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """ Splits the labeled dataframe into clusters

    Parameters:
    dataframe: Dataframe to be split

    Returns:
    data: Dict containing the cluster as key to the subset of the original dataframe
    containing only rows of that particular cluster
    """
    clusters = [*dataframe["cluster"].unique()]
    data = {}

    for cluster in clusters:
        data[cluster] = dataframe[dataframe["cluster"] == cluster]

    return data


def normalize_clusters_by_date(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Groups each cluster by the date

    Parameters:
    dataframe: Dataframe to be manipulated

    Returns:
    dataframe: Dataframe grouped by date
    """
    dataframe = dataframe[["date", "price"]]
    dataframe = dataframe.groupby("date").mean("price")

    return dataframe


def window_input_output(dataframe: pd.DataFrame, input_length: int, output_length: int) -> pd.DataFrame:
    """ Creates a sequence of observation to train the model

    Parameters:
    dataframe: Dataframe to be manipulated
    input_lenght: Number of training observations
    output_lenght: Number of labels

    Returns:
    dataframe: Dataframe with aritifical observations
    """
    df = dataframe.copy()
    
    for i in range(1, input_length):
        df[f"x_{i}"] = df["price"].shift(-i)


    for j in range(output_length):
        df[f"y_{j}"] = df["price"].shift(-output_length-j)
        
    df = df.dropna()
    return df

def xgboost_test(xgb_search: RandomizedSearchCV, X_test: pd.DataFrame, y_test: pd.DataFrame) -> np.ndarray:
    """ Testes the trained XGBoost against the data

    Parameters:
    xgb_search: The randomized search object
    X_test: The data to be tested
    y_test: The labels for each test data

    Returns:
    y_pred: The predicted values for each test data
    """
    y_pred = xgb_search.predict(X_test)
    print("Predicted RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

    return y_pred
