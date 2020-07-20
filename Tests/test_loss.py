import unittest

from Tests.Utils import create_C_W_X, create_C_W_X_d
from archive import objective_soft_max_old, objective_soft_max_gradient_W2, \
    objective_soft_max_gradient_W_old_but_gold
from loss_function import *
from numpy import array_equal, array


def f(X, W):
    return X.T @ W


class LossTests(unittest.TestCase):

    def test00_init_test(self):
        C, W, X = create_C_W_X()
        self.assertEqual(X.T.shape[1], W.shape[0])  # XT@W
        self.assertEqual(C.shape[1], X.shape[1])
        self.assertEqual(C.shape[0], W.shape[1])

    def test01_objective_soft_max_sanity_test(self):
        C, W, X = create_C_W_X()
        try:
            objective_soft_max(X, W, C)
            self.assertTrue(True)

        except:
            self.assertTrue(False)

    def test02_objective_soft_max_sanity_test(self):
        C, W, X, _, _ = create_C_W_X_d()
        c1 = np.round(objective_soft_max(X, W, C), 10)
        c2 = np.round(objective_soft_max_old(X, W, C), 10)

        self.assertEqual(c1, c2)

    def test03_objective_soft_max_gradient_W_sanity_test(self):
        C, W, X = create_C_W_X()

        try:
            objective_soft_max_gradient_W(X, W, C)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test04_objective_soft_max_gradient_W_compare_with_naive(self):
        C, W, X = create_C_W_X()
        c1 = np.round(objective_soft_max_gradient_W(X, W, C), 10)
        c2 = np.round(objective_soft_max_gradient_W2(X, W, C), 10)
        self.assertTrue(array_equal(c1, c2))

    def test05_objective_soft_max_gradient_W_dimension(self):
        C, W, X = create_C_W_X()
        exptected_shape = array([X.shape[0], W.shape[1]])
        result_shape = objective_soft_max_gradient_W(X, W, C).shape
        self.assertTrue(array_equal(exptected_shape, result_shape))

    def test06_objective_soft_max_gradient_X_sanity_test(self):
        C, W, X = create_C_W_X()
        try:
            objective_soft_max_gradient_X(X, W, C)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test07_objective_soft_max_gradient_X_dimension(self):
        C, W, X = create_C_W_X()
        exptected_shape = array([W.shape[0], X.shape[1]])
        result_shape = objective_soft_max_gradient_X(X, W, C).shape
        self.assertTrue(array_equal(exptected_shape, result_shape))

    def test08_objective_soft_max_gradient_W_dimension_compare_with_naive(self):
        C, W, X = create_C_W_X()
        c1 = np.round(objective_soft_max_gradient_W_old_but_gold(X, W, C), 10)
        c2 = np.round(objective_soft_max_gradient_W(X, W, C), 10)
        self.assertTrue(array_equal(c1, c2))


if __name__ == '__main__':
    unittest.main()
