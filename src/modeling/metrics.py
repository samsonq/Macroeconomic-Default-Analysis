"""

"""
import numpy as np
import math
from keras import backend as K


def recall(y_true, y_pred):
    """
    Calculates recall metric through false positive rates based on
    true values and predictions.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: recall score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision(y_true, y_pred):
    """
    Calculates precision metric through false negative rates based on
    true values and predictions.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: precision score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1(y_true, y_pred):
    """
    Calculates f1 score based on the precision and recall metrics of the
    true values and predictions.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: f1 score
    """
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))
