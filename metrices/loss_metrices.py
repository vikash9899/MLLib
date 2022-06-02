import numpy as np


def r2_square(actual, predict):
    corr_matrix = np.corrcoef(actual, predict)
    corr = corr_matrix[0, 1]
    R_sq = corr**2
    return round(R_sq, 3)

# Creating a custom function for MAE


def mean_absolute_error(y_true, predictions):
    return round(np.mean(np.abs(y_true - predictions)), 3)


def mean_square_error(y_test, pred):
    # Mean Squared Error
    MSE = np.square(np.subtract(y_test, pred)).mean()
    return round(MSE, 3)
