import numpy as np
from numpy import diag, log, sum, max, divide
from numpy.core._multiarray_umath import exp


def find_eta(X, W, XT_W=None):
    if XT_W is None:
        XT_W = X.T @ W
    return max(XT_W, axis=1).reshape(XT_W.shape[0], 1)


def objective_soft_max(X, W, C, WT_X=None):
    """
    :param X:
    :param W: shape #features x #labels
    :param C: shape #labels x #samples
    :return:
    """
    l, m = C.shape
    objective_value = 0
    if WT_X is None:
        XT_W = X.T @ W
    else:
        XT_W = WT_X.T
    eta = find_eta(X, W, XT_W)
    XT_W -= eta
    weighted_sum = sum(exp(XT_W), axis=1)
    for k in range(l):
        objective_value = objective_value + C[k, :].T @ log(divide(exp(XT_W[:, k]), weighted_sum))
    return - objective_value / m


def objective_soft_max_gradient_X(X, W, C, WT_X=None):
    if WT_X is None:
        WT_X = W.T @ X
    m = C.shape[1]
    return (1 / m) * W @ (divide(exp(WT_X), sum(exp(WT_X), axis=0)) - C)


def objective_soft_max_gradient_W(X, W, C, WT_X=None):
    m = C.shape[1]
    if WT_X is None:
        XT_W = X.T @ W
    else:
        XT_W = WT_X.T
    eta = find_eta(X, W, XT_W)
    XT_W -= eta
    weighted_sum = sum(exp(XT_W), axis=1)
    gradient = (1 / m) * X @ (divide(exp(XT_W), weighted_sum.reshape(weighted_sum.shape[0], 1)) - C.T)
    return gradient
