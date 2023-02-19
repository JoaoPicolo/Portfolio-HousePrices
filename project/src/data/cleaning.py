import numpy as np
import pandas as pd

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
