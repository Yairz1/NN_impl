import unittest

from Utils import create_C_W_X_d
from numpy import average, round, trace
from numpy.linalg import norm
from numpy.random import randn
from function import *
from loss_function import objective_soft_max, objective_soft_max_gradient_W, objective_soft_max_gradient_X


class test_gradient(unittest.TestCase):
    # ________________ loss tests __________________
    def test00_linear_convergence_loss_gradient_test_W(self):
        max_iter = 25
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, W, X, d, _ = create_C_W_X_d()
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            losses.append(abs(objective_soft_max(X, W + eps * d, C) -
                              objective_soft_max(X, W, C)))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.40 <= avg_val <= 0.6, msg=f'avg value = {avg_val}')
        # todo add  plot

    def test01_linear_convergence_loss_gradient_test_X(self):
        max_iter = 17
        C, W, X, _, d = create_C_W_X_d()
        eps0 = 0.5
        losses = []
        loss_convergence = []
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = round(objective_soft_max(X + eps * d, W, C), 10)
            objective2 = round(objective_soft_max(X, W, C), 10)
            losses.append(abs(objective1 - objective2))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.4 <= avg_val <= 0.6, msg=f'avg value = {avg_val}')
        # todo add  plot

    def test02_quadratic_convergence_loss_gradient_W(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, W, X, d, _ = create_C_W_X_d()
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = objective_soft_max(X, W + eps * d, C)
            objective2 = objective_soft_max(X, W, C)
            objective3 = eps * trace(d.T @ objective_soft_max_gradient_W(X, W, C))

            losses.append(abs(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')
        # todo add  plot

    def test03_quadratic_convergence_loss_gradient_X(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, W, X, _, d = create_C_W_X_d()
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = objective_soft_max(X + eps * d, W, C)
            objective2 = objective_soft_max(X, W, C)
            objective3 = eps * trace(d.T @ objective_soft_max_gradient_X(X, W, C))

            losses.append(abs(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')
        # todo add  plot

    # ________________ Specific layer linear tests __________________

    def test04_linear_convergence_gradient_test_W(self):
        max_iter = 25
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(X.shape[0], X.shape[0])
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            losses.append(norm(layer_function(X, W + eps * d, b) -
                               layer_function(X, W, b)))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.40 <= avg_val <= 0.6, msg=f'avg value = {avg_val}')
        # todo add  plot

    def test05_linear_convergence_gradient_test_X(self):
        max_iter = 25
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(*X.shape)
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            losses.append(norm(layer_function(X + eps * d, W, b) -
                               layer_function(X, W, b)))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.40 <= avg_val <= 0.6, msg=f'avg value = {avg_val}')
        # todo add  plot

    def test06_linear_convergence_gradient_test_b(self):
        max_iter = 25
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(*b.shape)
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            losses.append(norm(layer_function(X, W, b + eps * d) -
                               layer_function(X, W, b)))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.40 <= avg_val <= 0.6, msg=f'avg value = {avg_val}')
        # todo add  plot

    # ___________________________________________________________
    # ________________ Specific layer quadratic tests __________________
    def test07_quadratic_convergence_gradient_W(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(X.shape[0], X.shape[0])
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = layer_function(X, W + eps * d, b).T.reshape(-1, 1)
            objective2 = layer_function(X, W, b).T.reshape(-1, 1)
            objective3 = layer_function.jacMV_W(X, W, b, (eps * d).reshape(-1, 1),
                                                tanh_grad)  # (X, W, b, V, activation_grad)

            losses.append(norm(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')
        # todo add  plot

    def test07_quadratic_convergence_gradient_X(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(*X.shape)
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = layer_function(X + eps * d, W, b).T.reshape(-1, 1)
            objective2 = layer_function(X, W, b).T.reshape(-1, 1)
            V = (eps * d).T.reshape(-1, 1)
            objective3 = layer_function.jacMV_X(X, W, b, V, tanh_grad)  # (X, W, b, V, activation_grad)

            losses.append(norm(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')
        # todo add  plot

    def test08_quadratic_convergence_gradient_b(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(*b.shape)
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = layer_function(X, W, b + eps * d).T.reshape(-1, 1)
            objective2 = layer_function(X, W, b).T.reshape(-1, 1)
            V = (eps * repmat(d, 1, X.shape[1])).T.reshape(-1, 1)
            objective3 = layer_function.jacMV_b(X, W, b, V, tanh_grad)  # (X, W, b, V, activation_grad)

            losses.append(norm(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')
        # todo add  plot


if __name__ == '__main__':
    unittest.main()
