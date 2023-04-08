import pandas as pd

from data.utils import load_data_to_dataframe
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
        dataframe, _ = get_clusters(dataframe, n_clusters=5, ignore_cols=["date"])
        clusters = get_dataframes_by_cluster(dataframe)
        _ = get_forecast_by_cluster(clusters)


if __name__ == "__main__":
    main()
