import unittest

from Tests.Utils import data_factory
from activation import *
from numpy import array, ones, hstack, sum, average, round
from numpy.linalg import norm
from numpy.random import randn, rand
from matplotlib.pyplot import plot

from loss_function import objective_soft_max


class test_gradient(unittest.TestCase):

    def create_C_W_X_d(self, bias=True):
        _, C_val, _, X_val = data_factory('swiss', bias)
        W = randn(X_val.shape[0], C_val.shape[0])
        d = rand(X_val.shape[0])
        d[-1] = 0
        d = d.reshape(X_val.shape[0], 1)
        d_w = hstack([d] * W.shape[1])
        d_x = hstack([d] * X_val.shape[1])
        d_x = randn(*X_val.shape)
        return C_val, W, X_val, d_w, d_x

    def test_linear_convergence_gradient_test_W(self):
        max_iter = 25
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, W, X, d, _ = self.create_C_W_X_d()
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            losses.append(abs(objective_soft_max(X, W + eps * d, C) -
                              objective_soft_max(X, W, C)))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.48 <= avg_val <= 0.52, msg=f'ans = {avg_val}')

    def test_linear_convergence_gradient_test_X(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, W, X, _, d = self.create_C_W_X_d()
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            losses.append(abs(objective_soft_max(X + eps * d, W, C) -
                              objective_soft_max(X, W, C)))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.45 <= avg_val <= 0.55, msg=f'ans = {avg_val}')
