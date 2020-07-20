import numpy as np
from numpy import sum
from numpy.core._multiarray_umath import exp, log, divide

from loss_function import find_eta


def objective_soft_max_new_but_bad(X, W, C):
    """
    :param X:
    :param W: shape #features x #labels
    :param C: shape #labels x #samples
    :return:
    """
    l = C.shape[0]
    m = X.shape[1]
    objective_value = 0
    eta = find_eta(X, W)
    XT_W = X.T @ W - eta
    weighted_sum = sum(exp(XT_W), axis=1)
    for k in range(l):
        objective_value = objective_value + C[k, :].T @ log(divide(exp(XT_W[:, k]), weighted_sum))
    return - objective_value / m


def objective_soft_max_gradient_Wp(X, W, C, p):
    m = X.shape[1]
    eta = find_eta(X, W)
    XT_W = X.T @ W - eta
    weighted_sum = sum(exp(XT_W), axis=1)
    return (1 / m) * X @ (divide(exp(XT_W[:, p]), weighted_sum) - C[p, :])


def objective_soft_max_gradient_W2(X, W, C):
    l = C.shape[0]
    n = X.shape[0]
    m = X.shape[1]

    eta = find_eta(X, W)
    XT_W = X.T @ W - eta
    weighted_sum = sum(exp(XT_W), axis=1)
    grad = np.zeros((n, l))
    for p in range(l):
        grad[:, p] = (1 / m) * X @ (divide(exp(XT_W[:, p]), weighted_sum) - C[p, :])
    return grad


def objective_soft_max_gradient_W_old_but_gold(X, W, C):
    m = X.shape[1]
    eta = find_eta(X, W)
    XT_W = X.T @ W - eta
    weighted_sum = sum(exp(XT_W), axis=1)
    gradient = (1 / m) * X @ (divide(exp(XT_W), weighted_sum.reshape(weighted_sum.shape[0], 1)) - C.T)
    return gradient
