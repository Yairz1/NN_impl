import unittest

from Tests.Utils import create_C_W_X_d, data_factory
from loss_function import objective_soft_max_gradient_W, objective_soft_max
from optimizer import SGD
from matplotlib.pyplot import plot, semilogy, semilogx
from numpy import zeros, average, argmax
from numpy.random import randn
import matplotlib.pyplot as plt


def accuracy(X, W, C):
    XT_W = X.T @ W
    res = argmax(XT_W, axis=1)
    expected = argmax(C, axis=0)
    diff = res - expected
    return len(diff[diff == 0]) / len(diff)


class test_gradient(unittest.TestCase):

    def test00_batch_creation(self):
        m = 52
        _sgd = SGD(batch_size=10, m=m)
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

    def test02_sgd_sanity_with_epochs(self):
        C, W0, X, _, _ = create_C_W_X_d()
        optimizer = SGD(batch_size=256, m=X.shape[1])
        W = W0.copy()
        for epoch in range(15):
            W = optimizer.optimize(W, X, C,
                                   None,
                                   objective_soft_max_gradient_W, lr=1)

        self.assertTrue(True)

    def test03_sgd_sanity_with_epochs_with_momentum(self):
        C, W0, X, _, _ = create_C_W_X_d()
        optimizer = SGD(batch_size=256, m=X.shape[1])
        W = W0.copy()
        for epoch in range(15):
            W = optimizer.optimize(W, X, C,
                                   None,
                                   objective_soft_max_gradient_W, lr=1, momentum=0.001)

        self.assertTrue(True)

    # todo: should be in clear place subsection 3
    # todo write new function with the name train, gets batch and lr,epoch,momentum and dataset as parameter
    def test04_swiss_train(self):
        epochs = 50
        C_train, C_val, X_train, X_val = data_factory('GMMData')  # options: 'swiss','PeaksData','GMMData'
        W0 = randn(X_train.shape[0], C_train.shape[0])
        m, n = W0.shape
        W = W0.copy()
        optimizer = SGD(batch_size=256, m=X_train.shape[1])

        # ----------------- init stats lists -----------------
        W_history = zeros((W.shape[0] * W.shape[1], epochs))
        val_score = []
        train_score = []
        train_acc = []
        val_acc = []
        # ----------------------------------------------------

        for epoch in range(epochs):
            W = optimizer.optimize(W, X_train, C_train,
                                   objective_soft_max,
                                   objective_soft_max_gradient_W, lr=0.1, momentum=0)

            W_history[:, epoch] = W.reshape(W.shape[0] * W.shape[1])
            train_score.append(objective_soft_max(X_train, W, C_train))
            val_score.append(objective_soft_max(X_val, W, C_val))
            train_acc.append(accuracy(X_train, W, C_train))
            val_acc.append(accuracy(X_val, W, C_val))

        W_res = average(W_history, axis=1).reshape(m, n)
        train_score.append(objective_soft_max(X_train, W_res, C_train))
        val_score.append(objective_soft_max(X_val, W_res, C_val))
        # todo add plot epoch \ accuracy (wrote in train)
        plot(range(len(train_score)), train_score)
        self.assertTrue(True)
