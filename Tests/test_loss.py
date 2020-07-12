import unittest
from loss_function import *


def f(X, W):
    return X.T @ W


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


