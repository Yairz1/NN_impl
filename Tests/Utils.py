import numpy
import numpy as np
from numpy import diag, exp, log, divide, sum, ones, vstack, hstack
from numpy.core._multiarray_umath import array
from numpy.random.mtrand import randn, rand
from scipy.io import loadmat
import os
from loss_function import find_eta


def create_C_W_X():
    X = np.random.randn(5, 3)
    X = vstack((X, ones((1, 3))))
    C = array([[1, 0, 1],
               [0, 1, 0]])

    W = np.random.randn(6, 2)

    return C, W, X


def data_factory(data_name, bias=True):
    path_dic = {'swiss': '../data/NNdata/SwissRollData.mat'}

    if data_name == 'swiss':
        data = loadmat(path_dic['swiss'])
        Yt, Yv = data['Yt'], data['Yv']
        if bias:
            Yt = vstack((Yt, ones((1, Yt.shape[1]))))
            Yv = vstack((Yv, ones((1, Yv.shape[1]))))
        return data['Ct'], data['Cv'], Yt, Yv


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
    grad = np.zeros((l, n))
    for p in range(l):
        grad[p, :] = (1 / m) * X @ (divide(exp(XT_W[:, p]), weighted_sum) - C[p, :])
    return grad

def objective_soft_max_gradient_W3(X, W, C):
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


def create_C_W_X_d(bias=True):
    _, C_val, _, X_val = data_factory('swiss', bias)
    W = randn(X_val.shape[0], C_val.shape[0])
    d = rand(X_val.shape[0])
    d[-1] = 0
    d = d.reshape(X_val.shape[0], 1)
    d_w = hstack([d] * W.shape[1])
    d_x = hstack([d] * X_val.shape[1])
    d_x = randn(*X_val.shape)
    return C_val, W, X_val, d_w, d_x