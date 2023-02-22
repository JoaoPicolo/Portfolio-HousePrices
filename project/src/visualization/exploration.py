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


def get_outliers_plot(dataframe: pd.DataFrame, ignore_columns: list[str] = []):
    """ Shows a boxplot for each columns in the dataframe

    Parameters:
    dataframe: Dataframe on which the operation will be performed
    ignore_columns: Columns to ignored when creating the boxplots
    """
    cleaned_df = dataframe.drop(ignore_columns, axis=1)
    columns = cleaned_df.columns

    ncols = 2
    nrows = math.ceil(len(columns)/2)
    _, axes = plt.subplots(nrows, ncols, figsize=(20, 40))
    for index, axis in enumerate(axes.reshape(-1)):
        if index < len(columns):
            plot = sns.boxplot(y=cleaned_df[columns[index]], ax=axis)
            plot.set_title(f"{columns[index]} Values")
            plot.set_ylabel("")
            plot.ticklabel_format(style='plain', axis='y')
        else:  # If number of plots is odd, don't print the last one since it's expecting an even number of plots
            axis.axis('off')

    plt.tight_layout()  # adjusting the spacing between subplots
    plt.show()  # displaying the final figure


def get_distribution_plot(dataframe: pd.DataFrame, ignore_columns: list[str] = []):
    """ Shows a distribution plot for each columns in the dataframe

    Parameters:
    dataframe: Dataframe on which the operation will be performed
    ignore_columns: Columns to ignored when creating the distribution plots
    """
    cleaned_df = dataframe.drop(ignore_columns, axis=1)
    columns = cleaned_df.columns

    ncols = 2
    nrows = math.ceil(len(columns)/2)
    _, axes = plt.subplots(nrows, ncols, figsize=(20, 40))
    for index, axis in enumerate(axes.reshape(-1)):
        if index < len(columns):
            plot = sns.histplot(x=cleaned_df[columns[index]], ax=axis)
            plot.set_title(f"{columns[index]} Distribution")
            plot.ticklabel_format(style='plain', axis='x')
        else:  # If number of plots is odd, don't print the last one since it's expecting an even number of plots
            axis.axis('off')

    plt.tight_layout()  # adjusting the spacing between subplots
    plt.show  # displaying the final figure
