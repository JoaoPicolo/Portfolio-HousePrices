from typing import Dict, List, Union

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor

# TODO: Add tests for these method
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


def get_forecast_by_cluster(clusters: Dict[str, pd.DataFrame]) -> Dict[str, List]:
    """ gets the forecast model for each cluster

    Parameters:
    clusters: Dict containg the cluster as key to dataframe

    Returns:
    results: Dict containing the cluster as key to array of test values
    and predicted values
    """
    results = {}
    for key in clusters:
        cluster_df = clusters[key]
        cluster_df = normalize_clusters_by_date(cluster_df)
        seq_df = window_input_output(cluster_df, 5, 5)

        # Builds train and test set
        X_cols = [col for col in seq_df.columns if col.startswith("x")]
        X_cols.insert(0, "price")
        y_cols = [col for col in seq_df.columns if col.startswith("y")]

        # Will use all, but the two last rows for train
        X_train = seq_df[X_cols][:-2].values
        y_train = seq_df[y_cols][:-2].values

        # Will use the last two rows for test
        X_test = seq_df[X_cols][-2:].values
        y_test = seq_df[y_cols][-2:].values

        dt_seq = DecisionTreeRegressor(random_state=42)
        dt_seq.fit(X_train, y_train)
        dt_seq_preds = dt_seq.predict(X_test)

        results[key] = []
        results[key].append(y_test[1])
        results[key].append(dt_seq_preds[1])

    return results
