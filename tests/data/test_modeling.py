import sys

import pytest
import pandas as pd

sys.path.append(".")
from src.data.modeling import calculate_house_age, calculate_last_renovation


def test_house_age():
    """
        Test if the age function is calculating correctly
    """
    dataframe = pd.read_csv("data/interim/kc_house_data.csv")

    df_date = 2015
    dataframe["age"] = calculate_house_age(
        year_of_contruction=dataframe["yr_built"], current_year=df_date)
    sample = dataframe.sample()
    assert sample["age"].iloc[0] == df_date - sample["yr_built"].iloc[0]


def test_house_renovation():
    """
        Test if the age function is calculating correctly
    """
    dataframe = pd.read_csv("data/interim/kc_house_data.csv")

    df_date = 2015
    dataframe["age"] = calculate_house_age(
        year_of_contruction=dataframe["yr_built"], current_year=df_date)
    dataframe["last_renovation"] = calculate_last_renovation(
        year_of_renovation=dataframe["yr_renovated"], house_age=dataframe["age"], current_year=df_date)
    sample = dataframe.sample(n=1)

    if sample["yr_renovated"].iloc[0] == 0:
        assert sample["last_renovation"].iloc[0] == sample["age"].iloc[0]
    else:
        assert sample["last_renovation"].iloc[0] == df_date - sample["yr_renovated"].iloc[0]
