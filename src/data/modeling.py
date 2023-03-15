import pandas as pd


def calculate_house_age(year_of_contruction: pd.Series, current_year: int) -> pd.Series:
    """ Returns the age of the house

    Parameters:
    year_of_contruction: Series containing the construction year for each house
    current_year: Year on which the dataset was built

    Returns:
    age: Returns the age series
    """
    age = current_year - year_of_contruction

    return age


def calculate_last_renovation(year_of_renovation: pd.Series, house_age: pd.Series, current_year: int) -> pd.Series:
    """ Returns the years since the last renovation

    Parameters:
    year_of_renovation: Series containing the year of the last renovation. Zero if it was not renovated
    house_age: Series containing the age of each house
    current_year: Year on which the dataset was built

    Returns:
    last_renovation: Returns the Series containing the years since last renovation
    """

    last_renovation = []
    for idx, year in year_of_renovation.items():
        if year > 0:
            last_renovation.append(current_year - year)
        else: # If there was no renovation, the year is equal to the construction year
            last_renovation.append(house_age[idx])

    return pd.Series(last_renovation)