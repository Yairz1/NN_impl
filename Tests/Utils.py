import numpy as np
from numpy import ones, vstack, hstack
from numpy.core._multiarray_umath import array
from numpy.random.mtrand import randn, rand
from scipy.io import loadmat


def create_C_W_X():
    X = np.random.randn(5, 3)
    X = vstack((X, ones((1, 3))))
    C = array([[1, 0, 1],
               [0, 1, 0]])

    W = np.random.randn(6, 2)

    return C, W, X


def normlize_data(X, X_mean=None, X_std=None):
    if X_mean is None and X_std is None:
        X_mean = X.mean(axis=1).reshape(X.shape[0], 1)
        X_std = X.std(axis=1).reshape(X.shape[0], 1) + 0.1
    return (X - X_mean) / X_std, X_mean, X_std


def data_factory(data_name, bias=True):
    path_dic = {'swiss': '../data/NNdata/SwissRollData.mat'}

    if data_name == 'swiss':
        data = loadmat(path_dic['swiss'])
        Yt, Yv = data['Yt'], data['Yv']
        if bias:
            # -------- Train ----------
            Yt = vstack((Yt, ones((1, Yt.shape[1]))))
            Yt, X_mean, X_std = normlize_data(Yt)
            # -------- Validation ----------
            Yv = vstack((Yv, ones((1, Yv.shape[1]))))
            Yv, _, _ = normlize_data(Yv, X_mean, X_std)
        return data['Ct'], data['Cv'], Yt, Yv


def create_C_W_X_d(bias=True):
    _, C_val, _, X_val = data_factory('swiss', bias)
    W = randn(X_val.shape[0], C_val.shape[0])
    d = rand(X_val.shape[0])
    d[-1] = 0
    d = d.reshape(X_val.shape[0], 1)
    d_w = hstack([d] * W.shape[1])
    # d_x = hstack([d] * X_val.shape[1])
    d_x = randn(*X_val.shape)
    return C_val, W, X_val, d_w, d_x
