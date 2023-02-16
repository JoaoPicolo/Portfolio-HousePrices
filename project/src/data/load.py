import pandas as pd

def load_data_to_dataframe(data_path: str) -> pd.DataFrame:
    """ Loads the .csv path into a pandas dataframe

    Parameters:
    data_path: Path to the .csv file

    Returns:
    dataframe: Returns the created pandas dataframe
    """
    dataframe = pd.read_csv(data_path)
    return dataframe
