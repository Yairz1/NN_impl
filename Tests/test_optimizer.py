import unittest

from NeuralNetwork import NeuralNetwork
from Utils import create_C_W_X_d, data_factory, show_and_save_plot
from function import Function, tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b, tanh_grad, ReLU_F, \
    ReLU_grad
from loss_function import objective_soft_max_gradient_W, objective_soft_max
from optimizer import SGD
from function import Function
from matplotlib.pyplot import plot
from numpy import zeros, average, argmax
from numpy.random import randn

from train import train


class test_optimizer(unittest.TestCase):

    def test00_batch_creation(self):
        m = 52
        _sgd = SGD(batch_size=50, m=m)
        batches = _sgd.create_batches()
        all = []
        for batch in batches:
            all += batch
        for i in range(m):
            if i not in all:
                print(i)
                self.assertTrue(False)
        self.assertTrue(True)

    def test01_sgd_sanity(self):
        C, W0, X, _, _ = create_C_W_X_d()
        optimizer = SGD(batch_size=256, m=X.shape[1])
        W = optimizer.optimize(W0, X, C,
                               objective_soft_max,
                               objective_soft_max_gradient_W)
        self.assertTrue(True)

    def test02_sgd_sanity_with_epochs(self):
        C, W0, X, _, _ = create_C_W_X_d()
        optimizer = SGD(batch_size=256, m=X.shape[1])
        W = W0.copy()
        for epoch in range(15):
            W = optimizer.optimize(W, X, C,
                                   objective_soft_max,
                                   objective_soft_max_gradient_W, lr=1)

        self.assertTrue(True)

    def test03_sgd_sanity_with_epochs_with_momentum(self):
        C, W0, X, _, _ = create_C_W_X_d()
        optimizer = SGD(batch_size=256, m=X.shape[1])
        W = W0.copy()
        for epoch in range(15):
            W = optimizer.optimize(W, X, C,
                                   objective_soft_max,
                                   objective_soft_max_gradient_W, lr=1, momentum=0.001)

        self.assertTrue(True)

    # subsection 3
    def test04_PeaksData_train(self):
        data_name = 'PeaksData'
        C_train, C_val, X_train, X_val = data_factory(data_name)  # options: 'Swiss','PeaksData','GMMData'
        batch_size = 256
        epochs = 15
        lr = 0.1
        momentum = 0
        W0 = randn(X_train.shape[0], C_train.shape[0])
        train_score, train_acc, val_score, val_acc = train(objective_soft_max,
                                                           objective_soft_max_gradient_W,
                                                           False,  # isNeural
                                                           C_train,
                                                           C_val,
                                                           X_train,
                                                           X_val,
                                                           W0,
                                                           batch_size,
                                                           epochs,
                                                           lr,
                                                           momentum=momentum)
        show_and_save_plot(range(len(train_acc)), train_acc, range(len(val_acc)), val_acc,
                           title=f'Question 3 {data_name} new accuracy', semilog=True)
        print(f'{train_acc} {val_acc}')
    def test045_swiss_train(self):
        data_name = 'GMMData'
        C_train, C_val, X_train, X_val = data_factory(data_name)  # options: 'Swiss','PeaksData','GMMData'
        batch_size = 50
        epochs = 200
        lr = 0.01
        momentum = 0
        W0 = randn(X_train.shape[0], C_train.shape[0])
        train_score, train_acc, val_score, val_acc = train(objective_soft_max,
                                                           objective_soft_max_gradient_W,
                                                           False,  # isNeural
                                                           C_train,
                                                           C_val,
                                                           X_train,
                                                           X_val,
                                                           W0,
                                                           batch_size,
                                                           epochs,
                                                           lr,
                                                           momentum=momentum)
        show_and_save_plot(range(len(train_acc)), train_acc, range(len(val_acc)), val_acc,
                           title=f'Question 3 {data_name} new accuracy', semilog=True)
        print(f'{max(train_acc)} {max(val_acc)}')

    def test05_train_momentum(self):
        data_name = 'GMMData'
        C_train, C_val, X_train, X_val = data_factory(data_name)  # options: 'Swiss','PeaksData','GMMData'
        batch_size = 256
        epochs = 15
        lr = 0.1
        momentum = 0.1
        W0 = randn(X_train.shape[0], C_train.shape[0])

        train_score, train_acc, val_score, val_acc = train(objective_soft_max,
                                                           objective_soft_max_gradient_W,
                                                           False,  # isNeural
                                                           C_train,
                                                           C_val,
                                                           X_train,
                                                           X_val,
                                                           W0,
                                                           batch_size,
                                                           epochs,
                                                           lr,
                                                           momentum=momentum)
        show_and_save_plot(range(len(train_acc)), train_acc, range(len(val_acc)), val_acc,
                           title=f'{data_name} accuracy + sgd momentum')

