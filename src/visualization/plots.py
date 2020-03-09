"""
Visualize the features of the data to understand distributions and discover associations. Univariate
and bivariate analysis are important to fully understand the data. Trends may be discovered between
economic factors like GDP and default rate within a certain time period.
"""
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns


def single_barplot(data, x, y, hue=None):
    """
    Creates a bar plot to compare the values of two categories, stratified
    by another category (if applicable).

    :param data: dataset to visualize
    :param x: x-label of plot
    :param y: y-label of plot
    :param hue: extra condition to stratify in plot
    :return: barplot of inputted categories
    """
    ax = sns.barplot(x=x, y=y, hue=hue, data=data)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(y + " based on " + x)
    plt.show()
    return ax
