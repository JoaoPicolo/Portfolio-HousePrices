import numpy as np
import pandas as pd


def feature_selection(dataframe: pd.DataFrame, main_variable: str) -> pd.DataFrame:
    """ Returns a new dataframe without irrelevant features

    This method uses Pearson's correlation to remove any variables that
    are not linearly correlated to the main variable.

    Parameters:
    dataframe: Dataframe on which the operation will be performed
    main_variable: Variable on which each feature will be analyzed

    Returns:
    dataframe: Returns the cleaned dataframe
    """

    # Returns a dataframe of correlation for each column
    correlations = dataframe.corr(method="pearson", numeric_only=True)

    drop_columns = []

    main_series = correlations[main_variable]
    for index, value in main_series.items():
        if value <= 0.0:  # Will remove columns with negative correlation
            drop_columns.append(index)

    cleaned_df = dataframe.drop(drop_columns, axis=1)
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
