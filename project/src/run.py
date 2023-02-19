from data.cleaning import grade_formatting
from data.utils import load_data_to_dataframe, save_data_to_csv
from visualization.exploration import get_outliers_plot, get_correlation_plot


def main():
    try:
        dataframe = load_data_to_dataframe(
            data_path="../data/raw/kc_house_data.csv")
    except:
        print("It was not possible to read the provided .csv file")
        exit(0)

    get_outliers_plot(dataframe, drop_columns=["id", "date", "waterfront", "lat", "long", "zipcode", "yr_renovated"])
    # cleaned_df = grade_formatting(dataframe)
    # save_data_to_csv(cleaned_df, data_path="../data/interim/kc_house_data.csv")


if __name__ == "__main__":
    main()
