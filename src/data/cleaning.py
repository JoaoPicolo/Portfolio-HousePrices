import numpy as np
import pandas as pd


def drop_outliers_iqr(dataframe: pd.DataFrame, ignore_columns: list[str] = []) -> pd.DataFrame:
    """ Returns a dataframe without outliers

    Uses IQR approach to remove outliers.

    Parameters:
    dataframe: Dataframe on which the operation will be performed
    ignore_columns: Columns to ignored when looking for outliers

    Returns:
    dataframe: Returns the new dataframe without outliers
    """
    cols = [c for c in dataframe.columns if c not in ignore_columns]
    q1 = dataframe[cols].quantile(0.25)
    q3 = dataframe[cols].quantile(0.75)
    iqr = q3-q1

    condition = ~(
        (dataframe[cols] < (q1 - 1.5 * iqr)
         ) | (dataframe[cols] > (q3 + 1.5 * iqr))
    ).any(axis=1)

    cleaned_df = dataframe[condition]
    return cleaned_df


def grade_formatting(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Returns a formatted dataframe

    This formats the grade to be an integer instead of a range.
    If the grade is less or equal to 4 it will become zero, if it is
    between 5 and 9 (included) it will become 1, and if it is more or qual
    to 10 it will become two.

    Parameters:
    dataframe: Dataframe on which the operation will be performed

    Returns:
    dataframe: Returns the formatted dataframe
    """
    cleaned_df = dataframe.copy()

    cleaned_df["grade"] = np.where(
        cleaned_df["grade"] <= 4, 0, cleaned_df["grade"])
    cleaned_df["grade"] = np.where((cleaned_df["grade"] >= 5) & (
        cleaned_df["grade"] <= 9), 1, cleaned_df["grade"])
    cleaned_df["grade"] = np.where(
        cleaned_df["grade"] >= 10, 2, cleaned_df["grade"])
    return cleaned_df
