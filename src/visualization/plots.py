"""
Visualize the features of the data to understand distributions and discover associations. Univariate
and bivariate analysis are important to fully understand the data. Trends may be discovered between
economic factors like GDP and default rate within a certain time period.
"""
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns


def barplot(data, x, y, hue=None):
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


def histogram(column, data, normal=True, kde=False, bins=None):
    """
    Plots a histogram to visualize the distribution of a numerical
    column in the inputted data to find skewness.

    :param column: numerical column in data
    :param data: data to plot
    :param normal: normalize histogram
    :param kde: boolean of fitting kernel density estimate
    :param bins: bins of histogram
    :return: histogram of numerical column
    """
    ax = sns.distplot(data[column].dropna(), norm_hist=normal, kde=kde, bins=bins)
    plt.title("Distribution of " + column)
    return ax


def scatterplot(x, y, data, hue=None, regression=False):
    """
    This function returns a seaborn bar plot based on the data columns passed in.

    :param x: x-axis column as a string
    :param y: y-axis column as a string
    :param data: dataframe containing above columns
    :param hue: hue column as a string
    :param regression: boolean of whether to plot regression
    :returns: scatter plot of the columns
    """
    if not regression:
        return sns.relplot(x=x, y=y, hue=hue, data=data)
    else:
        assert hue is None, "Can't have Hue with Regression Plot"
        return sns.regplot(x=x, y=y, data=data)


def count_distribution(x, data, size=(14, 6)):
    """
    Plots and returns a count plot that displays the counts of each
    category within a given column.

    :param x: categorical column to display counts
    :param data: dataframe containing columns
    :param size: size of plot
    :return: count plot of x
    """
    plt.figure(figsize=size)

    g = sns.countplot(x=x, data=data)
    g.set_title(x+" Distribution", fontsize=15)
    g.set_xlabel(x, fontsize=13)
    g.set_ylabel('Count', fontsize=13)

    proportions = []
    for status in g.patches:
        height = status.get_height()
        proportions.append(height)
        g.text(status.get_x() + status.get_width() / 2.,
               height + 3,
               "{:1.2f}%".format(height / len(data) * 100),
               ha="center", fontsize=12)

    g.set_ylim(0, max(proportions) * 1.1)
