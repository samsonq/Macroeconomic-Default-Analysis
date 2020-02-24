"""

"""
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def single_barplot(data, x, y, hue=None):
    """

    :param data:
    :param x:
    :param y:
    :param hue:
    :return:
    """
    ax = sns.barplot(x=x, y=y, hue=hue, data=data)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(y + " based on " + x)
    plt.show()
    return ax
