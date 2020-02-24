"""
Prepare training, validation, and testing data after preprocessing of the large dataset. Used in
training and evaluating models.
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def feature_selection(data, features):
    """
    Choose which features to use for training.

    :param data: preprocessed dataset
    :param features: list of features to use
    :return: data with selected features
    """
    return data[features]


def prepare_data(data, label="loan_status", valid_split=0.2, test_split=0.3):
    """
    Splits and returns the training and validation sets for the data.

    :param data: preprocessed dataset
    :param label: label of data
    :param valid_split: percentage to use as validation data
    :param test_split: percentage to use as test data
    :returns: training, validation, testing sets
    """
    X_train = data.drop(label, axis=1)  # define training features set
    y_train = data[label]  # define training label set
    # use part of the data as testing data
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_split, random_state=0)
    # use part of the training data as validation data
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_split, random_state=0)
    return X_train, X_valid, X_test, y_train, y_valid, y_test
