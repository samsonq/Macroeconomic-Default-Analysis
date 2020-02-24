"""
Perform ETL on Lending Club dataset. Methods to pull, clean, and reformat data.
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
    Drops the inputted column from the data either by row or column.

    :param data: data to drop
    :param col: column to drop null
    :param method: drop by row or column
    :return: data with dropped nulls
    """
    null = list(data[pd.isnull(data[col])].index)
    axis = 0 if method == "row" else 1
    return data.drop(null, axis=axis)


def encode_categories(data, columns):
    """
    Encodes categorical data values into numerical values.

    :param data: data with columns
    :param columns: list of columns to encode
    :return: encoded dataframe
    """
    encoded = data.copy()
    le = LabelEncoder()
    for column in columns:
        le.fit(data[column])
        encoded[column] = le.transform(data[column])
    return encoded


def scale(data, columns, method="standard"):
    """
    Scales the inputted columns either through standardization or
    min/max values.

    :param data: data with columns
    :param columns: list of columns to scale
    :param method: way of scaling
    :return: scaled datafarme
    """
    scaled = data.copy()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    for column in columns:
        col_array = np.array(data[column]).reshape(-1, 1)
        scaled[column] = scaler.fit_transform(col_array)
    return scaled
