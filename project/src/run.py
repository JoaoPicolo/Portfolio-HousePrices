from data.load import load_data_to_dataframe

def main():
    try:
        dataframe = load_data_to_dataframe("../data/raw/kc_house_dataa.csv")
    except:
        print("It was not possible to read the provided .csv file")
        exit(0)

    print(dataframe.head())

if __name__ == "__main__":
    main()