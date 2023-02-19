from data.cleaning import grade_formatting
from data.utils import load_data_to_dataframe, save_data_to_csv


def main():
    try:
        dataframe = load_data_to_dataframe(
            data_path="../data/raw/kc_house_data.csv")
    except:
        print("It was not possible to read the provided .csv file")
        exit(0)

    cleaned_df = grade_formatting(dataframe)
    save_data_to_csv(cleaned_df, data_path="../data/interim/kc_house_data.csv")


if __name__ == "__main__":
    main()
