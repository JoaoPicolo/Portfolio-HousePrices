import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def get_dataframe_types() -> dict[str, np.dtype]:
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
