import numpy as np
from sklearn.model_selection import RandomizedSearchCV

# TODO: Add tests for this method
def evaluate_cv_search(search: RandomizedSearchCV):
    print("Best parameters:", search.best_params_)
    print("Lowest RMSE: ", np.sqrt(-search.best_score_))
    print("Feature importance:", search.best_estimator_.feature_importances_)
