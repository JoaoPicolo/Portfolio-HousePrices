from data.cleaning import grade_formatting, drop_outliers_iqr
from data.utils import load_data_to_dataframe, save_data_to_csv


def main():
    try:
        dataframe = load_data_to_dataframe(
            data_path="../data/raw/kc_house_data.csv")
    except:
        print("It was not possible to read the provided .csv file")
        exit(0)

    dataframe = grade_formatting(dataframe)

    ignore_cols = ["id", "date", "waterfront",
                   "lat", "long", "zipcode", "yr_renovated"]
    df_no_outlier = drop_outliers_iqr(dataframe, ignore_columns=ignore_cols)

    # Save new dataframes
    save_data_to_csv(dataframe, data_path="../data/interim/kc_house_data.csv")
    save_data_to_csv(dataframe=df_no_outlier, data_path="../data/interim/kc_house_data_no_outlier.csv")


if __name__ == "__main__":
    main()
