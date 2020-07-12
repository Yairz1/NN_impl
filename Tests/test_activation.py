import unittest
from activation import *
from numpy import array


class ActivationTests(unittest.TestCase):

    def test00_softmax(self):
        x = array([5, 5, 5, 5])
        self.assertTrue(np.array_equal(softmax(x), array([0.25] * 4)), msg="wrong softmax result")

    def test01_ReLU(self):
        x = array([5, -5, -5, 5])
        self.assertTrue(np.array_equal(ReLU(x), array([5, 0, 0, 5])), msg="ReLU softmax result")
