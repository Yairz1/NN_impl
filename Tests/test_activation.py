import unittest

from activation import *
from numpy import array


class ActivationTests(unittest.TestCase):

    def test01_ReLU(self):
        x = array([5, -5, -5, 5])
        self.assertTrue(np.array_equal(ReLU(x), array([5, 0, 0, 5])), msg="ReLU softmax result")


if __name__ == '__main__':
    unittest.main()
