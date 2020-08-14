import numpy as np
from numpy import ndarray, kron, identity, vstack, zeros, diag, block, tanh
from numpy.matlib import repmat


class Function:
    def __init__(self, f, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b):
        self.f = f
        self.jacMV_X = jacMV_X
        self.jacMV_W = jacMV_W
        self.jacMV_b = jacMV_b
        self.jacTMV_X = jacTMV_X
        self.jacTMV_W = jacTMV_W
        self.jacTMV_b = jacTMV_b

    def apply(self, X, W, b):
        return self.f(X, W, b)

    def __call__(self, X, W, b):
        return self.apply(X, W, b)


def ReLU(x: ndarray):
    x_copy = x.copy()
    x_copy[x_copy < 0] = 0
    return x_copy


def ReLU_F(X, W, b):
    assert b.shape == (W.T.shape[0], 1), "should reshape bias to be (n,1)"
    return ReLU(W.T @ X + b)


def tanh_F(X, W, b):
    assert b.shape == (W.T.shape[0], 1), "should reshape bias to be (n,1)"
    return tanh(W.T @ X + b)


def ReLU_grad(x):
    x_copy = x.copy()
    x_copy[x_copy <= 0] = 0
    x_copy[x_copy > 0] = 1
    return x_copy


def tanh_grad(x):
    return 1 - np.power(tanh(x), 2)


# ----------------------------------------------------

def jacMV_X(X, W, b, V, activation_grad):
    return (activation_grad(W.T @ X + b) * (W.T @ V.reshape(X.shape[1], -1).T)).T.reshape(-1, 1)


# v

def jacTMV_X(X, W, b, V, activation_grad):
    # return (W @ (activation_grad(W.T @ X + b) * repmat(V, 1, X.shape[1]))).T.reshape(-1)
    return (W @ (activation_grad(W.T @ X + b) * V.reshape(X.shape[1], -1).T)).T.reshape(-1, 1)


# v

# ----------------------------------------------------

def jacMV_W(X, W, b, V, activation_grad):
    return diag(activation_grad(W.T @ X + b).T.reshape(-1)) @ kron(X.T, identity(W.shape[1])) @ V


# v

def jacTMV_W(X, W, b, V, activation_grad):
    return (diag(activation_grad(W.T @ X + b).T.reshape(-1)) @ kron(X.T, identity(W.shape[1]))).T @ V


# ----------------------------------------------------

def jacMV_b(X, W, b, V, activation_grad):
    V = repmat(V, 1, X.shape[1]).T.reshape(-1, 1)
    return activation_grad(W.T @ X + b).T.reshape(-1, 1) * V


def jacTMV_b(X, W, b, V, activation_grad):
    return block([diag(col) for col in activation_grad(W.T @ X + b).T]) @ V

# ----------------------------------------------------
# W = np.array([[1, 2, 3],
#               [4, 5, 6]]).T
#
# X = np.array([[1, 2, 3, 4, 5],
#               [4, 5, 6, 4, 5],
#               [1, 1, 1, 4, 5]])
#
# V = np.array([[0.5],
#               [0.7],
#               [0.3]])
#
# V2 = np.array([[0.5],
#                [0.7],
#                ])
#
# V3 = np.array([[0.1],
#                [0.2],
#                [0.3],
#                [0.4],
#                [0.5],
#                [0.6], ])
#
# V4 = np.array([[0.1],
#                [0.2],
#                [0.3],
#                [0.4],
#                [0.5],
#                [0.6],
#                [0.7],
#                [0.8],
#                [0.9],
#                [1], ])
#
# V5 = np.array([[0.1],
#                [0.2]])
# b = np.ones((2, 1))
# activation_grad = tanh_grad
#
# print(jacMV_X(X, W, b, V, activation_grad))
# print(jacTMV_X(X, W, b, V2, activation_grad))
#
# print(jacMV_W(X, W, b, V3, activation_grad))
# print(jacTMV_W(X, W, b, V4, activation_grad))
#
# print(jacMV_b(X, W, b, V5, activation_grad))
# print(jacTMV_b(X, W, b, V4, activation_grad))
