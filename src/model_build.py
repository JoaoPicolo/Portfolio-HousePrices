from models.utils import evaluate_cv_search
from models.model_build import xgboost_train, xgboost_test
from data.utils import load_data_to_dataframe, get_train_test_data

from visualization.analysis import plot_prediction

def main():
    datasets = ["../data/processed/kc_house_data.csv",
                "../data/processed/kc_house_data_no_outlier.csv"]

    for dataset in datasets:
        try:
            dataframe = load_data_to_dataframe(data_path=dataset)
        except:
            print("It was not possible to read the provided .csv file")
            exit(0)

        print(f"\n\nAnalysing {dataset.split('/')[3]}")


        X_train, X_test, y_train, y_test = get_train_test_data(dataframe, ["price"])

       
        xgboost_search = xgboost_train(X_train, y_train, n_iter=10)
        evaluate_cv_search(xgboost_search)
        y_pred = xgboost_test(xgboost_search, X_test, y_test)
        plot_prediction(y_test, y_pred)


if __name__ == "__main__":
    main()
