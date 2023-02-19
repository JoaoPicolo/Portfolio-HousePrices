import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_correlation_plot(dataframe: pd.DataFrame):
    """ Shows a correlation matrix for the dataframe

    Parameters:
    dataframe: Dataframe on which the operation will be performed
    """
    sns.heatmap(dataframe.corr(method="pearson"), annot=True)
    plt.title("Pearson's correlation coefficients")
    plt.show()


def get_outliers_plot(dataframe: pd.DataFrame, drop_columns: list[str]):
    """ Shows a boxplot for each columns in the dataframe

    Parameters:
    dataframe: Dataframe on which the operation will be performed
    drop_columns: Columns to ignore when creating the boxplots
    """
    cleaned_df = dataframe.drop(drop_columns, axis=1)
    columns = cleaned_df.columns

    ncols = 2
    nrows = math.ceil(len(columns)/2)
    _, axes = plt.subplots(nrows, ncols, figsize=(20,40)) # creating a figure with 2 rows and 2 columns of plots
    for index, col in enumerate(columns):
        ax = axes[math.floor(index/2), index%2] # selecting the current subplot
        plot = sns.boxplot(y=cleaned_df[col], ax=ax)
        plot.set_title(f"{col} Distribution")
        plot.set_ylabel("")
        plot.ticklabel_format(style='plain', axis='y')

    plt.tight_layout() # adjusting the spacing between subplots
    plt.show() # displaying the final figure

