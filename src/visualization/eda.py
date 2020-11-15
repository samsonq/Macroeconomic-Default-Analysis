"""
Functions for performing EDA on the LendingClub and economy data. It is useful to observe
statistics about the data like max, min, and mean to understand distribution, as well as
look for outliers and missing values.
"""
import numpy as np
import pandas as pd


def outlier(arr):
    """

    :param arr:
    :return:
    """
    return


def print_statistics(arr, population=False):
    """
    Computes and prints statistics/parameters from the inputted data.

    :param arr: array of data
    :param population: population or sample data
    """
    print("Max: ", max(arr))
    print("Min: ", min(arr))
    print("Mean: ", np.mean(arr))
    if population:
        print("Standard Deviation: ", np.std(arr))
    else:
        print("Standard Deviation: ", np.std(arr, ddof=1))
