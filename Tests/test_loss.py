import unittest

from Tests.Utils import objective_soft_max_old, objective_soft_max_gradient_W2
from loss_function import *
from numpy import array_equal, ones, hstack, vstack, zeros


def f(X, W):
    return X.T @ W


class LossTests(unittest.TestCase):
    def create_X_C_W(self):
        X = np.random.randn(5, 3)
        X = np.vstack((X, ones((1, 3))))
        C = array([[1, 0, 1],
                   [0, 1, 0]])

        W = np.random.randn(6, 2)

        return C, W, X

    def test00_init_test(self):
        C, W, X = self.create_X_C_W()
        self.assertEqual(X.T.shape[1], W.shape[0])  # XT@W
        self.assertEqual(C.shape[1], X.shape[1])
        self.assertEqual(C.shape[0], W.shape[1])

    def test01_objective_soft_max_sanity_test(self):
        C, W, X = self.create_X_C_W()
        try:
            objective_soft_max(X, W, C)
            self.assertTrue(True)

        except:
            self.assertTrue(False)

    def test02_objective_soft_max_sanity_test(self):
        C, W, X = self.create_X_C_W()
        c1 = np.round(objective_soft_max(X, W, C), 10)
        c2 = np.round(objective_soft_max_old(X, W, C), 10)

        self.assertEqual(c1, c2)

    def test03_objective_soft_max_gradient_W_sanity_test(self):
        C, W, X = self.create_X_C_W()

        try:
            objective_soft_max_gradient_W(X, W, C)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test04_objective_soft_max_gradient_W_compare_with_naive(self):
        C, W, X = self.create_X_C_W()
        c1 = np.round(objective_soft_max_gradient_W(X, W, C), 10)
        c2 = np.round(objective_soft_max_gradient_W2(X, W, C), 10)
        self.assertTrue(array_equal(c1, c2))

    def test05_objective_soft_max_gradient_W_dimension(self):
        C, W, X = self.create_X_C_W()
        exptected_shape = array([W.shape[1], X.shape[0]])
        result_shape = objective_soft_max_gradient_W(X, W, C).shape
        self.assertTrue(array_equal(exptected_shape, result_shape))

    def test06_objective_soft_max_gradient_X_sanity_test(self):
        C, W, X = self.create_X_C_W()
        try:
            objective_soft_max_gradient_X(X, W, C)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test07_objective_soft_max_gradient_X_dimension(self):
        C, W, X = self.create_X_C_W()
        exptected_shape = array([W.shape[0], X.shape[1]])
        result_shape = objective_soft_max_gradient_X(X, W, C).shape
        self.assertTrue(array_equal(exptected_shape, result_shape))
