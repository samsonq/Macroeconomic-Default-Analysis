"""
Build and train various types of classification models to predict default/loan status of
a borrower, given feature data about the loan.

Models:
    - Neural Network Classifier
    - Support Vector Machine (SVM)
    - Decision Tree & Random Forest (ensemble)
    - Logistic Regression
    - K Nearest Neighbors
    - Naive Bayes
"""
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from metrics import recall, precision, f1


def train_and_evaluate(mdl, train_X, train_y, valid_X, valid_y):
    """
    Trains and evaluates a model based on training/validation data.

    :param mdl: model to fit
    :param train_X: training features
    :param train_y: training label
    :param valid_X: validation features
    :param valid_y: validation label
    :return: accuracy of model on validation set
    """
    models = {"svm": SVC(), "random_forest": RandomForestClassifier(), "log_reg": LogisticRegression(),
              "knn": KNeighborsClassifier(), "naive_bayes": GaussianNB(), "decision_tree": DecisionTreeClassifier()}
    model = models[mdl]
    model.fit(train_X, train_y)
    predictions = model.predict(valid_X)
    return model, accuracy_score(valid_y, predictions)


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
    classes = len(set(train_y))
    train_y = to_categorical(train_y, classes)
    valid_y = to_categorical(valid_y, classes)

    mdl = Sequential()
    mdl.add(Dense(64, activation="relu", input_dim=train_X.shape[1]))
    mdl.add(Dense(128, activation="relu"))

    dropout_rate = 0.2
    mdl.add(Dropout(dropout_rate))  # regularization by dropping weights

    mdl.add(Dense(classes, activation="softmax"))

    mdl.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", recall, precision, f1])

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


def predict(test, model, save=False):
    """
    Make predictions on the test set with the inputted data.

    :param test: testing set
    :param model: model to make predictions
    :param save: save predictions to new csv
    :return: testing set with predictions made
    """
    predictions = model.predict(test)
    preds = test.assign(Predictions=predictions)
    if save:
        preds.to_csv("preds.csv")
    return preds
