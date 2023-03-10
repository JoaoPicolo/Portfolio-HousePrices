import sys

import pytest
import pandas as pd

sys.path.append(".")
from src.data.cleaning import drop_outliers_iqr, grade_formatting

@pytest.fixture
def get_dataframe_missing() -> int:
    """
        Returns the number of missing values in the dataframe
    """
    dataframe = pd.read_csv("data/raw/kc_house_data.csv")
    return dataframe.isna().sum().sum()


def test_no_missing_values(get_dataframe_missing: callable):
    """
        Test if there are no missing values in the dataframe
    """
    assert get_dataframe_missing == 0


@pytest.fixture
def get_dataframe_duplicates() -> pd.DataFrame:
    """
        Returns the duplicated values in the dataframe
    """
    dataframe = pd.read_csv("data/raw/kc_house_data.csv")
    return dataframe[dataframe.duplicated()]


def test_no_missing_values(get_dataframe_duplicates: callable):
    """
        Test if there are no duplicates values in the dataframe
    """
    assert get_dataframe_duplicates.empty


def test_outliers_dropping():
    """
        Test if the dataframe is correctly bwing returned after dropping outliers
    """
    dataframe = pd.read_csv("data/raw/kc_house_data.csv")
    ignore_cols = ["id", "date", "waterfront",
                   "lat", "long", "zipcode", "yr_renovated"]
    dataframe = drop_outliers_iqr(dataframe, ignore_columns=ignore_cols)
    assert type(dataframe) == pd.DataFrame


def test_grade_data_formatting():
    """
        Test if the grade formatting is working
    """
    dataframe = pd.read_csv("data/raw/kc_house_data.csv")
    dataframe = grade_formatting(dataframe)
    assert set(dataframe["grade"]) == {0, 1, 2}
