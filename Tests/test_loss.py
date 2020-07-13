import unittest

from Tests.Utils import objective_soft_max_old, objective_soft_max_gradient_W2
from loss_function import *
from numpy import array_equal


def f(X, W):
    return X.T @ W



class LossTests(unittest.TestCase):

    def test00_objective_soft_max_sanity_test(self):
        C, W, X = self.create_X_C_W()
        try:
            objective_soft_max(X, W, C)
            self.assertTrue(True)

        except:
            self.assertTrue(False)

    def test01_objective_soft_max_sanity_test(self):
        C, W, X = self.create_X_C_W()
        c1 = np.round(objective_soft_max(X, W, C), 10)
        c2 = np.round(objective_soft_max_old(X, W, C), 10)

        self.assertEqual(c1, c2)

    def create_X_C_W(self):
        X = np.random.randn(5, 3)
        C = array([[1, 0],
                   [0, 1],
                   [1, 0]]).T
        W = np.random.randn(5, 2)
        return C, W, X

    def test02_objective_soft_max_gradient_W_sanity_test(self):
        C, W, X = self.create_X_C_W()

        try:
            objective_soft_max_gradient_W(X, W, C)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test03_objective_soft_max_gradient_W_compare_with_naive(self):
        C, W, X = self.create_X_C_W()
        c1 = np.round(objective_soft_max_gradient_W(X, W, C), 10)
        c2 = np.round(objective_soft_max_gradient_W2(X, W, C), 10)
        self.assertTrue(array_equal(c1, c2))
