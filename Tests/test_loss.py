import unittest
from loss_function import *


def f(X, W):
    return X.T @ W


def objective_soft_max_old(X, W, C):
    l = C.shape[1]
    m = X.shape[1]
    objective_value = 0
    eta = find_eta(X, W)
    XT_W = X.T @ W - eta
    weighted_sum = sum(exp(XT_W), axis=1)  # todo minus eta ?
    wighted_sum_matrix = diag(1 / weighted_sum)
    for k in range(l):
        objective_value = objective_value + C[:, k].T @ log(wighted_sum_matrix @ exp(XT_W[:, k]))
    return - objective_value / m

class LossTests(unittest.TestCase):

    def test00_objective_soft_max_sanity_test(self):
        X = np.random.randn(5, 3)
        C = array([[1, 0],
                   [0, 1],
                   [1, 0]])
        W = np.random.randn(5, 2)
        try:
            objective_soft_max(X, W, C)
            self.assertTrue(True)

        except:
            self.assertTrue(False)

    def test01_objective_soft_max_sanity_test(self):
        X = np.random.randn(5, 3)
        C = array([[1, 0],
                   [0, 1],
                   [1, 0]])
        W = np.random.randn(5, 2)
        c1 = np.round(objective_soft_max(X, W, C), 10)
        c2 = np.round(objective_soft_max_old(X, W, C), 10)

        self.assertEqual(c1, c2)
