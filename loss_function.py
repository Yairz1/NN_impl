import numpy as np
from numpy import exp, zeros, diag, log, array, sum, max, divide
import scipy
from activation import softmax


def find_eta(X, W):
    return max(X.T @ W, axis=0)


def objective_soft_max(X, W, C):
    l = C.shape[1]
    m = X.shape[1]
    objective_value = 0
    eta = find_eta(X, W)
    XT_W = X.T @ W - eta
    weighted_sum = sum(exp(XT_W), axis=1)  # todo minus eta ?
    for k in range(l):
        objective_value = objective_value + C[:, k].T @ log(divide(exp(XT_W[:, k]), weighted_sum))
    return - objective_value / m


def objective_soft_max_gradient_X(X, W, C):
    pass

def objective_soft_max_gradient_W(X, W, C):
    pass

