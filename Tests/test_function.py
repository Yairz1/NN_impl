import unittest

from function import *
from numpy import array, identity
from numpy.random import randn


class test_function(unittest.TestCase):

    def test00_ReLU(self):
        x = array([5, -5, -5, 5])
        self.assertTrue(np.array_equal(ReLU(x), array([5, 0, 0, 5])), msg="ReLU softmax result")

    def test01_tanh(self):
        x = identity(5)
        self.assertEqual(tanh(x).shape, x.shape)

    def test02_ReLU_F(self):
        x = identity(5)
        W = randn(5, 5)
        self.assertEqual(ReLU_F(x, W).shape, x.shape)

    def test03_ReLU_grad(self):
        x = identity(5)
        self.assertEqual(ReLU_grad(x).shape, x.shape)

    def test04_tanh_grad(self):
        x = identity(5)
        self.assertEqual(tanh_grad(x).shape, x.shape)

    def test05_f_grad_X_mul_V(self):
        x = identity(5)
        w = randn(5, 5)
        v = randn(5, 5)

        self.assertEqual(f_grad_X_mul_V(x, w, v, activation_grad=ReLU_grad).shape, (x.shape[0], v.shape[1]))
        self.assertEqual(f_grad_X_mul_V(x, w, v, activation_grad=tanh_grad).shape, (x.shape[0], v.shape[1]))

    def test06_f_grad_W_mul_V(self):
        x = identity(5)
        w = randn(5, 5)
        v = randn(5, 5)
        self.assertEqual(f_grad_W_mul_V(x, w, v, activation_grad=tanh_grad).shape, (x.shape[0], v.shape[1]))


if __name__ == '__main__':
    unittest.main()
