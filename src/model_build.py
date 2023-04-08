import pandas as pd

from models.utils import evaluate_cv_search
from models.model_build import xgboost_train, xgboost_test
from data.utils import load_data_to_dataframe, get_train_test_data
from models.utils import get_clusters, get_dataframes_by_cluster, get_forecast_by_cluster

def main():
    datasets = ["../data/processed/kc_house_data.csv",
                "../data/processed/kc_house_data_no_outlier.csv"]
    
    for dataset in datasets:
        try:
            dataframe = load_data_to_dataframe(data_path=dataset)
        except:
            print("It was not possible to read the provided .csv file")
            exit(0)

        # Cast column
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe = dataframe.sort_values(by="date")

        # Price forecast
        print("Price forescat:")
        dataframe, _ = get_clusters(dataframe, n_clusters=5, ignore_cols=["date"])
        clusters = get_dataframes_by_cluster(dataframe)
        _ = get_forecast_by_cluster(clusters)

        # Price prediction
        print("\nPrice prediction:")
        dataframe = load_data_to_dataframe(data_path=dataset)
        dataframe = dataframe.drop("date", axis="columns")
        X_train, X_test, y_train, y_test = get_train_test_data(dataframe, ["price"])
        xgboost_search = xgboost_train(X_train, y_train, n_iter=1)
        evaluate_cv_search(xgboost_search)
        xgboost_test(xgboost_search, X_test, y_test)



if __name__ == "__main__":
    main()
