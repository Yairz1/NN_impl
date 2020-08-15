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

    # todo: should be in clear place subsection 3
    # todo: added more options of b_size epochs and lr.
    def test04_swiss_train(self):
        data_name = 'GMMData'
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
                           title=f'{data_name} new accuracy')

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

    def test06_train_nn(self):
        data_name = 'PeaksData'
        C_train, C_val, X_train, X_val = data_factory(data_name)  # options: 'Swiss','PeaksData','GMMData'
        n = X_val.shape[0]
        l = C_val.shape[0]
        batch_size = 64
        epochs = 25
        lr = 0.1
        momentum = 0.01
        # layer_function, activation_grad = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W,
        #                                            jacTMV_b), tanh_grad
        layer_function, activation_grad = Function(ReLU_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b), ReLU_grad

        L = 3
        width = 20
        first_layer_dim = (n, width)
        layer_dim = (width, width)
        output_dim = (width, l)
        model = NeuralNetwork(num_of_layers=L, f=layer_function,
                              activation_grad=activation_grad,  # ReLU_grad,tanh_grad
                              first_layer_dim=first_layer_dim,
                              layer_dim=layer_dim,
                              output_dim=output_dim)
        params = model.params_to_vector(model.params).copy()
        train_score, train_acc, val_score, val_acc = train(model,
                                                           model.backward,
                                                           True,  # isNeural
                                                           C_train,
                                                           C_val,
                                                           X_train,
                                                           X_val,
                                                           params,  # W0
                                                           batch_size,
                                                           epochs,
                                                           lr,
                                                           momentum=momentum)
        show_and_save_plot(range(len(train_acc)), train_acc, range(len(val_acc)), val_acc,
                           title=f'DataSet:{data_name}| #layer = {L} | batch:{batch_size} epochs:{epochs} lr:{lr} ')
        print(f'train_acc {train_acc}')
        print(f'val_acc {val_acc}')
