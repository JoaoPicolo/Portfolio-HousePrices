import sys

import pytest
import numpy as np
import pandas as pd

from typing import Dict

sys.path.append(".")
from src.data.utils import load_data_to_dataframe, save_data_to_csv


@pytest.fixture
def get_dataframe_types() -> Dict[str, np.dtype]:
    """
        Loads the dataframe to be tested
    """
    dataframe = pd.read_csv("data/raw/kc_house_data.csv")
    return dataframe.dtypes.to_dict()


def test_df_data_types(get_dataframe_types: callable):
    """
        Tests for the data types in the dataframe
    """
    default_types = {
        "id": np.dtype("int64"), "date": np.dtype("O"), "price": np.dtype("float64"),
        "bedrooms": np.dtype("int64"), "bathrooms": np.dtype("float64"),
        "sqft_living": np.dtype("int64"), "sqft_lot": np.dtype("int64"),
        "floors": np.dtype("float64"), "waterfront": np.dtype("int64"),
        "view": np.dtype("int64"), "condition": np.dtype("int64"), "grade": np.dtype("int64"),
        "sqft_above": np.dtype("int64"), "sqft_basement": np.dtype("int64"),
        "yr_built": np.dtype("int64"), "yr_renovated": np.dtype("int64"),
        "zipcode": np.dtype("int64"), "lat": np.dtype("float64"), "long": np.dtype("float64"),
        "sqft_living15": np.dtype("int64"), "sqft_lot15": np.dtype("int64")
    }

    assert get_dataframe_types == default_types


def test_df_load():
    """
        Test if the dataframe is being correctly loaded
    """
    dataframe = load_data_to_dataframe("data/raw/kc_house_data.csv")
    assert type(dataframe) == pd.DataFrame

def test_df_save():
    """
        Test if the dataframe is being correctly saved
    """
    dataframe = pd.read_csv("data/raw/kc_house_data.csv")
    result = save_data_to_csv(dataframe, data_path="data/raw/kc_house_data.csv")
    
    assert result == None
