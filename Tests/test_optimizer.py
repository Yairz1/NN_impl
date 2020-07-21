import unittest

from Tests.Utils import create_C_W_X_d
from loss_function import objective_soft_max_gradient_W
from optimizer import SGD


class test_gradient(unittest.TestCase):

    def test00_batch_creation(self):
        m = 52
        _sgd = SGD(batch_size=10, epochs=300, m=m)
        batches = _sgd.create_batches()
        all = []
        for batch in batches:
            all += batch
        for i in range(m):
            if i not in all:
                self.assertTrue(False)
        self.assertTrue(True)

    def test01_sgd_sanity(self):
        C, W0, X, _, _ = create_C_W_X_d()
        optimizer = SGD(batch_size=256, m=X.shape[1])
        W = optimizer.optimize(W0, X, C,
                               None,
                               objective_soft_max_gradient_W)
        self.assertTrue(True)
