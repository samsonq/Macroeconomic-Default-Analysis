"""
Evaluation of models created: performance on test set, signs of overfitting, etc.
Hyperparameter tuning of models through processes like cross-validation.
"""
import numpy as np

import tensorflow as tf
import keras
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, accuracy_score


def predict(mdl, test_X, test_y):
    """

    :param mdl:
    :param test_X:
    :param test_y:
    :return:
    """
