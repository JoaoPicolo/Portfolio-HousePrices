from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error as mape

from utils import normalize_clusters_by_date, window_input_output

# TODO: Add tests for these method
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
        print(f"MAPE for cluster {key} is {mape(dt_seq_preds.reshape(1, -1), y_test.reshape(1, -1))}")

    return results

def xgboost_train(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int = 5) -> RandomizedSearchCV:
    """ Trains the XGBoost model against the data

    Parameters:
    X_train: The data to be used during training
    y_ttrain: The labels for each train data
    n_iter: Number of iterations to be used during parameters tuning in the model

    Returns:
    search: The randomized search object containing the tunned model
    """
    params = {
        'max_depth': [3, 5, 6, 10, 15, 20],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': np.arange(0.5, 1.0, 0.1),
        'colsample_bytree': np.arange(0.4, 1.0, 0.1),
        'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
        'n_estimators': [100, 500, 1000]
    }

    xgb_model = xgb.XGBRegressor(random_state=42)
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=params,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_iter=n_iter, verbose=2)

    search.fit(X_train, y_train)

    return search
