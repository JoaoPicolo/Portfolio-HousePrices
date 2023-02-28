from data.utils import load_data_to_dataframe
from visualization.exploration import get_correlation_plot, get_outliers_plot, get_distribution_plot


def main():
    try:
        dataframe = load_data_to_dataframe(
            data_path="../data/raw/kc_house_data.csv")
    except:
        print("It was not possible to read the provided .csv file")
        exit(0)

    ignore_cols = ["id", "date", "waterfront",
                   "lat", "long", "zipcode", "yr_renovated"]
    get_correlation_plot(dataframe)
    get_outliers_plot(dataframe, ignore_columns=ignore_cols)
    get_distribution_plot(dataframe, ignore_columns=ignore_cols)


if __name__ == "__main__":
    main()
