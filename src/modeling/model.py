"""

"""
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from metrics import recall, precision, f1

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def build_train_nn(train_X, train_y, valid_X, valid_y, epochs, batch_size):
    """
    Builds and trains a neural network based on the inputted data and parameters
    and evaluates it on the validation data.

    :param train_X: training features
    :param train_y: training label
    :param valid_X: validation features
    :param valid_y: validation label
    :param epochs: number of epochs to train for
    :param batch_size: batch size for each epoch
    :return: fitted neural network model
    """
    mdl = Sequential()
    mdl.add(Dense(64, activation="relu", input_dim=train_X.shape[1]))
    mdl.add(Dense(64, activation="relu"))

    dropout_rate = 0.2
    mdl.add(Dropout(dropout_rate))  # regularization by dropping weights

    mdl.add(Dense(1, activation="sigmoid"))
    mdl.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", recall, precision, f1])

    fit = mdl.fit(train_X, train_y, epochs=epochs,
                  batch_size=batch_size, verbose=1, validation_data=(valid_X, valid_y))

    plt.figure(figsize=(15, 10))
    plt.plot(fit.history["loss"], label="training")
    plt.plot(fit.history["val_loss"], label="validation")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc=7)
    plt.show()
    return mdl
