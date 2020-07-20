import unittest

from Tests.Utils import create_C_W_X_d
from numpy import average, round, trace

from loss_function import objective_soft_max, objective_soft_max_gradient_W, objective_soft_max_gradient_X


class test_gradient(unittest.TestCase):
    def test00_linear_convergence_gradient_test_W(self):
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
        self.assertTrue(0.48 <= avg_val <= 0.52, msg=f'ans = {avg_val}')

    def test01_linear_convergence_gradient_test_X(self):
        max_iter = 25
        passed = 0
        for times in range(5):
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
            if 0.45 <= avg_val <= 0.55:
                passed += 1
        self.assertTrue(passed >= 3, msg='error,failed more then 2 times')

    def test02_quadratic_convergence_gradient_W(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, W, X, d, _ = create_C_W_X_d()
        n, m = d.shape
        vec_d = d.copy().reshape(n * m, )
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = objective_soft_max(X, W + eps * d, C)
            objective2 = objective_soft_max(X, W, C)
            objective_val = objective_soft_max_gradient_W(X, W, C).reshape(n * m, )
            objective3 = eps * (vec_d.T @ objective_val)
            # objective3 = eps * trace(d.T @ objective_soft_max_gradient_W(X, W, C))

            losses.append(abs(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')


if __name__ == '__main__':
    unittest.main()
