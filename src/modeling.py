import pandas as pd

from data.utils import load_data_to_dataframe, save_data_to_csv
from data.modeling import calculate_house_age, calculate_last_renovation


def main():
    datasets = ["../data/interim/kc_house_data.csv",
                "../data/interim/kc_house_data_no_outlier.csv"]

    for dataset in datasets:
        try:
            dataframe = load_data_to_dataframe(data_path=dataset)
        except:
            print("It was not possible to read the provided .csv file")
            exit(0)

        # This dataframe is from 2015
        df_date = 2015

        # Calculates new features
        dataframe["age"] = calculate_house_age(
            year_of_contruction=dataframe["yr_built"], current_year=df_date)
        dataframe["last_renovation"] = calculate_last_renovation(
            year_of_renovation=dataframe["yr_renovated"], house_age=dataframe["age"], current_year=df_date)

        # Remove columns that do not affect the target variable
        ignore_columns = ["id", "yr_built", "yr_renovated",
                          "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]
        dataframe = dataframe.drop(labels=ignore_columns, axis="columns")

        # Formats the date column to contain month and year only
        dataframe["date"] = pd.to_datetime(dataframe["date"], format="%Y%m%dT%H%M%S")
        dataframe["date"] = dataframe["date"].dt.strftime("%m-%Y")

        # Save processed dataframe
        save_data_to_csv(dataframe, data_path=dataset.replace("interim", "processed"))

if __name__ == "__main__":
    main()
