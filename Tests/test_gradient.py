import unittest

from Utils import create_C_W_X_d
from numpy import average, round, trace, array
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

    def test007_quadratic_convergence_gradient_X(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        X = array([[-0.05680519, 0.04855737, -0.01294702, 0.22890241, -0.03285573, -1.04920292],
                   [0.94595808, 0.58875477, 0.74938208, 0.47105753, 0.88231689, -2.04920292],
                   [0.65439475, 1.06168175, 0.89133197, 1.33827674, 0.74290448, -3.04920292],
                   [-1.04920292, -0.98638338, -1.03932691, -1.03685701, -1.11946368, -4.04920292],
                   [-0.90285724, -0.40958798, -0.90025687, -0.97141045, -1.21387255, -5.04920292]])

        W = array([[0.62506598, 0.82961435, 0.5880866, 1.53119086, 1.16146083, 0.2671512],
             [1.13491018, -0.94388051, 0.49330541, 1.01477576, 0.5376924, 0.12150312],
             [-0.96057435, -0.58400606, 1.56789662, 0.13952425, 1.21025993, 0.1145395],
             [-1.48350759, 2.14568026, 0.21704896, -0.00658095, 1.34996381, -2.16069315],
             [-0.58935949, 0.41107469, -0.32381905, 0.27922415, -1.4647874, -0.18714347],])

        b = array([[0.2589714],
                   [0.14706977],
                   [-0.3933798],
                   [-1.25054223],
                   [-0.45488394],
                   [-0.2891693]]).reshape(-1, 1)

        d = array([[-1.33017472e-03, -8.02771650e-01, 1.46097145e+00, 2.33671768e+00, 7.88790151e-01, 1.88790151e-01],
                   [2.48529270, -7.16905013e-01, -4.84506807e-01, 2.40679447e-01, 1.02071847, 2.88790151e-01],
                   [4.33170511e-01, 4.63751757e-01, -1.37268999e-01, -1.46550989e-01, 4.61949104e-01, 3.88790151e-01],
                   [4.07699846e-01, 1.54921843e-01, -3.97222247e-01, -1.71962162e+00, 1.74330034e-02, 4.88790151e-01],
                   [-1.14063176e+00, -3.05619564e-01, 3.19862564e-01, 2.72153021e-01, 5.42045265e-01, 5.88790151e-01]])

        d2 = array([[-1.33017472e-03, -8.02771650e-01, 1.46097145e+00, 2.33671768e+00, 7.88790151e-01, 1.88790151e-01],
                   [2.48529270, -7.16905013e-01, -4.84506807e-01, 2.40679447e-01, 1.02071847, 2.88790151e-01],
                   [4.33170511e-01, 4.63751757e-01, -1.37268999e-01, -1.46550989e-01, 4.61949104e-01, 3.88790151e-01],
                   [4.07699846e-01, 1.54921843e-01, -3.97222247e-01, -1.71962162e+00, 1.74330034e-02, 4.88790151e-01],
                   [-1.14063176e+00, -3.05619564e-01, 3.19862564e-01, 2.72153021e-01, 5.42045265e-01, 5.88790151e-01],
                   [-1.14063176e+00, -3.05619564e-01, 3.19862564e-01, 2.72153021e-01, 5.42045265e-01, 5.88790151e-01]])

        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = layer_function(X + eps * d, W, b).T.reshape(-1, 1)
            objective2 = layer_function(X, W, b).T.reshape(-1, 1)
            V = d.T.reshape(-1, 1)
            V2 = d2.T.reshape(-1, 1)
            objective4 = layer_function.jacTMV_X(X, W, b, V2, tanh_grad)  # (X, W, b, V, activation_grad)
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
            objective3 = layer_function.jacMV_b(X, W, b, eps * d, tanh_grad)  # (X, W, b, V, activation_grad)

            losses.append(norm(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')
        # todo add  plot

    def test09_transpose_jacMVW(self):
        _, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        u = randn(X.shape[0] * X.shape[1]).reshape(-1, 1)
        v = randn(X.shape[0], X.shape[0]).reshape(-1, 1)

        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        jac_W_v = layer_function.jacMV_W(X, W, b, v, tanh_grad)  # (X, W, b, V, activation_grad)
        jac_T_W_u = layer_function.jacTMV_W(X, W, b, u, tanh_grad)  # (X, W, b, V, activation_grad)
        res = abs(u.T @ jac_W_v - v.T @ jac_T_W_u)
        self.assertTrue(res < 1e-5, msg=f'res = {res}')

    def test10_transpose_jacMVb(self):
        _, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        v = randn(*b.shape)
        u = randn(W.shape[1] * X.shape[1]).reshape(-1, 1)

        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        jac_b_v = layer_function.jacMV_b(X, W, b, v, tanh_grad)  # (X, W, b, V, activation_grad)
        jac_T_b_u = layer_function.jacTMV_b(X, W, b, u, tanh_grad)  # (X, W, b, V, activation_grad)
        res = abs(u.T @ jac_b_v - v.T @ jac_T_b_u)
        self.assertTrue(res < 1e-5, msg=f'res = {res}')

    def test11_transpose_jacMVX(self):
        _, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        v = randn(X.shape[0]*X.shape[1]).reshape(-1, 1)
        u = randn(X.shape[0] * X.shape[1]).reshape(-1, 1)

        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        jac_X_v = layer_function.jacMV_X(X, W, b, v, tanh_grad)  # (X, W, b, V, activation_grad)
        jac_T_X_u = layer_function.jacTMV_X(X, W, b, u, tanh_grad)  # (X, W, b, V, activation_grad)
        res = abs(u.T @ jac_X_v - v.T @ jac_T_X_u)
        self.assertTrue(res < 1e-5, msg=f'res = {res}')


if __name__ == '__main__':
    unittest.main()
