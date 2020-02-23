"""
Perform ETL on Lending Club dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def remove_header(path):
    """
    Removes annoying link header on data sets.
    :param path: path to data
    """
    with open(path, "r") as r:
        next_line = next(r)
        with open(".././data/LendingClub/datasets/Processed.csv", "w") as w:
            for line in r:
                w.write(line)


def drop_null(data, col="loan_status", method="row"):
    """

    :param data: data to drop
    :param col: column to drop null
    :param method: drop by row or column
    :return: data with dropped nulls
    """
    null = list(data[pd.isnull(data[col])].index)
    axis = 0 if method == "row" else 1
    return data.drop(null, axis=axis)


def encode_categories(col):
    """

    :param col: column to encode
    :return: encoded column
    """
    le = LabelEncoder()
    return le.fit_transform(col)


def scale(col, method="standard"):
    """

    :param col: column to scale
    :param method: way of scaling
    :return: scaled column
    """
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    return
