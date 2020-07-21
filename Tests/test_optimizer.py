import unittest

from Tests.Utils import create_C_W_X_d
from loss_function import objective_soft_max_gradient_W
from optimizer import SGD
from matplotlib.pyplot import plot, semilogy, semilogx
from numpy import zeros, average
import matplotlib.pyplot as plt


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

    def test01_sgd_sanity_with_epochs(self):
        C, W0, X, _, _ = create_C_W_X_d()
        optimizer = SGD(batch_size=256, m=X.shape[1])
        W = W0.copy()
        for epoch in range(15):
            W = optimizer.optimize(W, X, C,
                                   None,
                                   objective_soft_max_gradient_W, lr=1)

        self.assertTrue(True)

    def test02_sgd_sanity_with_epochs_with_momentum(self):
        C, W0, X, _, _ = create_C_W_X_d()
        optimizer = SGD(batch_size=256, m=X.shape[1])
        W = W0.copy()
        for epoch in range(15):
            W = optimizer.optimize(W, X, C,
                                   None,
                                   objective_soft_max_gradient_W, lr=1, momentum=0.001)

        self.assertTrue(True)

    def test03_swiss_train(self):
        epochs = 15
        C, W0, X, _, _ = create_C_W_X_d()
        m, n = W0.shape
        optimizer = SGD(batch_size=256, m=X.shape[1])
        W = W0.copy()
        W_history = zeros((W.shape[0] * W.shape[1], epochs + 1))
        for epoch in range(epochs):
            W = optimizer.optimize(W, X, C,
                                   None,
                                   objective_soft_max_gradient_W, lr=1, momentum=0.001)
            W_history[:, epoch] = W.reshape(W.shape[0] * W.shape[1])
        W_res = average(W_history, axis=1).reshape(m, n)  # 0 it's columns
        # todo add plot epoch \ accuracy (wrote in train)
        # plot(range(len(losses) - 1), loss_convergence)
        self.assertTrue(True)
