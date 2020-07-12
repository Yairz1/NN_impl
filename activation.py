import numpy as np
from numpy import exp, sum, ndarray


def softmax(x: ndarray):
    return exp(x) / sum(exp(x))


def ReLU(x: ndarray):
    x[x < 0] = 0
    return x
