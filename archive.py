import numpy as np
from numpy import sum,diag
from numpy.core._multiarray_umath import exp, log, divide

from loss_function import find_eta


# works good
def objective_soft_max_Wj(X, W, C):
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


# works good
def objective_soft_max_old(X, W, C):
    l = C.shape[0]
    m = X.shape[1]
    objective_value = 0
    eta = find_eta(X, W)
    XT_W = X.T @ W - eta
    weighted_sum = sum(exp(XT_W), axis=1)
    wighted_sum_matrix = diag(1 / weighted_sum)
    for k in range(l):
        objective_value = objective_value + C[k, :].T @ log(wighted_sum_matrix @ exp(XT_W[:, k]))
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
    l = C.shape[0]
    n = X.shape[0]
    m = X.shape[1]
    eta = find_eta(X, W)
    XT_W = X.T @ W - eta
    weighted_sum = sum(exp(XT_W), axis=1)
    wighted_sum_matrix = np.diag(1 / weighted_sum)
    grad = np.zeros((n, l))
    for p in range(l):
        grad[:, p] = (1 / m) * X @ (wighted_sum_matrix @ exp(XT_W[:, p]) - C[p, :])
    return grad
