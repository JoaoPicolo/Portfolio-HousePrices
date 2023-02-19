import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_correlation_plot(dataframe: pd.DataFrame):
    sns.heatmap(dataframe.corr(method="pearson"), annot=True)
    plt.title("Pearson's correlation coefficients")
    plt.show()
