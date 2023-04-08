import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# TODO: Add tests for these method
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