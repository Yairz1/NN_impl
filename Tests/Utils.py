from numpy import diag, exp, log, divide, sum
import numpy as np

from loss_function import find_eta


def objective_soft_max_old(X, W, C):
    l = C.shape[0]
    m = X.shape[1]
    objective_value = 0
    eta = find_eta(X, W)
    XT_W = X.T @ W - eta
    weighted_sum = sum(exp(XT_W), axis=1)  # todo minus eta ?
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
    grad = np.zeros((l, n))
    for p in range(l):
        grad[p, :] = (1 / m) * X @ (divide(exp(XT_W[:, p]), weighted_sum) - C[p, :])
    return grad
