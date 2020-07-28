import unittest

from Utils import create_C_W_X_d, data_factory, show_and_save_plot
from loss_function import objective_soft_max_gradient_W, objective_soft_max
from optimizer import SGD
from matplotlib.pyplot import plot
from numpy import zeros, average, argmax
from numpy.random import randn

from train import train


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
    def test04_swiss_train(self):
        data_name = 'GMMData'
        C_train, C_val, X_train, X_val = data_factory(data_name)  # options: 'swiss','PeaksData','GMMData'
        batch_size = 256
        epochs = 15
        lr = 0.1
        momentum = 0
        train_score, train_acc, val_score, val_acc = train(C_train,
                                                           C_val,
                                                           X_train,
                                                           X_val,
                                                           batch_size,
                                                           epochs,
                                                           lr,
                                                           momentum=momentum)
        show_and_save_plot(range(len(train_acc)), train_acc, range(len(val_acc)), val_acc,
                           title=f'{data_name} accuracy')
