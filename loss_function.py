import numpy as np
from numpy import exp, zeros, diag, log, array, sum
import scipy
from activation import softmax


def objective_soft_max(X, W, C):
    l = C.shape[1]
    m = X.shape[1]
    objective_value = 0
    weighted_sum = sum(exp(X.T @ W), axis=1)  # todo minus eta ?
    wighted_sum_matrix = diag(1 / weighted_sum)
    for k in range(l):
        objective_value = objective_value + C[:, k].T @ log(wighted_sum_matrix @ exp(X.T @ W[:, k]))
    return - objective_value / m


def objective_soft_max_gradient(X, W, C):
    x = 1
