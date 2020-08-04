import numpy as np
from numpy import exp, sum, ndarray, sinh, cosh, diag, kron, identity, exp


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
    return cosh(x) ** -2


def f_grad_X_mul_V(X, W, V, activation_grad):
    return W[:-1, :] @ activation_grad(W.T @ X) @ V


def f_grad_W_mul_V(X, W, V, activation_grad):
    return (activation_grad(W.T @ X) * V) @ X.T


class Function:
    def __init__(self, f, gradient_x, gradient_w):
        self.f = f
        self.gradient_x = gradient_x
        self.gradient_w = gradient_w

    def apply(self, X, W):
        return self.f(X, W)

    def gradient_x(self, X, W):
        return self.gradient_x(X, W)

    def gradient_w(self, X, W):
        return self.gradient_w(X, W)

    def __call__(self, X, W):
        return self.apply(X, W)
