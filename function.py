import numpy as np
from numpy import exp, sum, ndarray, sinh, cosh, diag, kron, identity, exp, vstack, zeros


def ReLU(x: ndarray):
    x_copy = x.copy()
    x_copy[x_copy < 0] = 0
    return x_copy


def tanh(x: ndarray):
    return exp(x) - exp(-x) / exp(x) + exp(-x)


def ReLU_F(X, W):
    return ReLU(W.T @ X)


def ReLU_grad(x):
    x_copy = x.copy()
    x_copy[x_copy <= 0] = 0
    x_copy[x_copy > 0] = 1
    return x_copy


def tanh_grad(x):
    return 1 - np.power(tanh(x), 2)


def f_grad_X_mul_V(X, W, V, activation_grad):
    res = W[:-1, :] @ (activation_grad(W.T @ X) * V)  # avoid the bias row
    return vstack([res, zeros(res.shape[1])])


def f_grad_W_mul_V(X, W, V, activation_grad):
    return (activation_grad(W.T @ X) * V) @ X.T


class Function:
    def __init__(self, f, f_grad_X_mul_V, f_grad_W_mul_V):
        self.f = f
        self.f_grad_X_mul_V = f_grad_X_mul_V
        self.f_grad_W_mul_V = f_grad_W_mul_V

    def apply(self, X, W):
        return self.f(X, W)

    def __call__(self, X, W):
        return self.apply(X, W)
