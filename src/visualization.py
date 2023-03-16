from data.utils import load_data_to_dataframe
from visualization.exploration import get_correlation_plot, get_outliers_plot, get_distribution_plot, get_scatter_plot


def main():
    try:
        dataframe = load_data_to_dataframe(
            data_path="../data/raw/kc_house_data.csv")
    except:
        print("It was not possible to read the provided .csv file")
        exit(0)

    # Generates plots 
    get_correlation_plot(dataframe)
    get_outliers_plot(dataframe)
    get_distribution_plot(dataframe)
    get_scatter_plot(dataframe, target_variable=dataframe["price"])


if __name__ == "__main__":
    main()
