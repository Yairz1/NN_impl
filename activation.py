import numpy as np
from numpy import exp, sum, ndarray, sinh, cosh


def ReLU(x: ndarray):
    x[x < 0] = 0
    return x


def tanh(x: ndarray):
    return sinh(x) / cosh(x)
