import numpy as np
from numpy import exp, zeros, diag, log, array, sum, max, divide
import scipy
from activation import softmax


def find_eta(X, W):
    return max(X.T @ W, axis=0)

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


def objective_soft_max(X, W, C):
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



def objective_soft_max_gradient_X(X, W, C):
    WT_X = W.T @ X
    m = X.shape[1]
    return (1 / m) * W @ (divide(exp(WT_X), sum(exp(WT_X), axis=0)) - C)


# todo : add another objective_soft_max_gradient_X with naive impl and compare results.


def objective_soft_max_gradient_W(X, W, C):
    m = X.shape[1]
    eta = find_eta(X, W)
    XT_W = X.T @ W - eta
    weighted_sum = sum(exp(XT_W), axis=1)
    gradient = (1 / m) * X @ (divide(exp(XT_W), weighted_sum.reshape(weighted_sum.shape[0], 1)) - C.T)
    return gradient.T


